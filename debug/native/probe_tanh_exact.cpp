/*
 * Can the tanh inside situ be replaced by a branch-free formula?
 *
 * situ (comfy/lib/xpu_int8/src.cpp:situ_reg) calls sycl::tanh twice per element.
 * On saturated inputs (|x| > ~9) the default tanh is 3.5x slower than on small
 * ones -- see probe_ffn_situ: 0.71 ms vs 0.09 ms for the whole activation at
 * M=3974, K=8192. The slowness is data-dependent, so it is the implementation's
 * slow path (overflow/denormal handling), not the transcendental itself.
 *
 * The replacement is safe to judge exhaustively because BOTH arguments are
 * bf16 before tanh:
 *   tanh(g/4)  -- g is bf16 and /4 is exact, so the argument is bf16(g)/4
 *   tanh(h/25) -- h/25 is explicitly rounded to bf16 first
 * so tanh only ever sees a bf16 value. There are 65536 of them. If a candidate
 * reproduces sycl::tanh's *bf16-rounded* result for all 65536, it is bit-for-bit
 * as good as the shipped kernel on any input the model can produce.
 *
 *   ./build.sh && ./bin/probe_tanh_exact
 *   ./bin/probe_tanh_exact --time
 */
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float, 16>;
using I32 = vec<int32_t, 16>;

static inline float bf_round(float x) {
    return float(BF(x));
}

// candidate: value of tanh, computed branch-free
template <int C>
static inline float cand(float x) {
    if constexpr (C == 0) {
        return sycl::tanh(x);
    } else if constexpr (C == 1) {
        // (e-1)/(e+1) with e = 2^(2x*log2e), magnitude first
        const float a = sycl::fmin(sycl::fabs(x), 20.0f);
        const float e = sycl::exp2(2.8853900817779268f * a);
        return sycl::copysign((e - 1.0f) / (e + 1.0f), x);
    } else if constexpr (C == 2) {
        const float a = sycl::fmin(sycl::fabs(x), 20.0f);
        const float e = sycl::exp(2.0f * a);
        return sycl::copysign((e - 1.0f) / (e + 1.0f), x);
    } else if constexpr (C == 3) {
        const float a = sycl::fmin(sycl::fabs(x), 20.0f);
        const float m = sycl::expm1(2.0f * a);
        return sycl::copysign(m / (m + 2.0f), x);
    } else if constexpr (C == 4) {
        const float a = sycl::fmin(sycl::fabs(x), 20.0f);
        const float e = sycl::exp2(2.8853900817779268f * a);  // 2/log(2)
        return sycl::copysign(1.0f - 2.0f / (e + 1.0f), x);
    } else {
        // clamp + the shipped tanh: exact by construction, skips the slow path
        if (sycl::fabs(x) > 9.011f) return sycl::copysign(1.0f, x);
        return sycl::tanh(x);
    }
}

int main(int argc, char **argv) {
    queue q;
    const bool do_time = argc > 1 && std::strcmp(argv[1], "--time") == 0;

    // ---- exactness over every bf16 input -------------------------------
    {
        constexpr int N = 65536;
        uint16_t *in = malloc_shared<uint16_t>(N, q);
        float *out = malloc_shared<float>(N * 6, q);
        for (int i = 0; i < N; i++) in[i] = uint16_t(i);
        q.parallel_for(range<1>(N), [=](id<1> i) {
            const float x = float(sycl::bit_cast<BF>(in[i]));
            out[i] = cand<0>(x);
            out[N + i] = cand<1>(x);
            out[2 * N + i] = cand<2>(x);
            out[3 * N + i] = cand<3>(x);
            out[4 * N + i] = cand<4>(x);
            out[5 * N + i] = cand<5>(x);
        }).wait();

        const char *names[6] = {"sycl::tanh (reference)", "exp2 (e-1)/(e+1)",
                                "exp (e-1)/(e+1)", "expm1 m/(m+2)",
                                "exp2 1-2/(e+1)", "clamp>9.011 + sycl::tanh"};
        std::printf("exactness over all 65536 bf16 inputs (compare bf16-rounded):\n");
        for (int c = 0; c < 6; c++) {
            long bad_bf = 0, bad_fp = 0, bad_nan = 0;
            double maxrel = 0;
            for (int i = 0; i < N; i++) {
                const float r = out[i], v = out[c * N + i];
                if (std::isnan(v)) { bad_nan++; continue; }
                if (v != r) bad_fp++;
                const float rb = bf_round(r), vb = bf_round(v);
                if (std::memcmp(&rb, &vb, 4) != 0) bad_bf++;
                if (r != 0 && std::isfinite(r))
                    maxrel = std::max(maxrel, std::fabs(double(v) - r) / std::fabs(double(r)));
            }
            std::printf("  %-28s bf16 mismatch %6ld / 65536   !=fp32 %6ld   NaN %ld   max rel %.2e\n",
                        names[c], bad_bf, bad_fp, bad_nan, maxrel);
        }
    }

    if (!do_time) return 0;

    // ---- speed on a large array, two distributions ----------------------
    const size_t N = 1 << 22;
    float *x = malloc_shared<float>(N, q);
    float *y = malloc_shared<float>(N, q);
    auto fill = [&](bool sat) {
        uint32_t rng = 7u;
        for (size_t i = 0; i < N; i++) {
            rng = rng * 1103515245u + 12345u;
            const float u = float(rng & 0xFFFFFF) / 16777215.0f;  // [0,1)
            if (sat) x[i] = (u * 2.0f - 1.0f) * 60.0f;           // saturated
            else x[i] = (u * 2.0f - 1.0f) * 0.7f;                 // typical
        }
        q.wait();
    };
    auto bench = [&](const char *nm, auto kernel) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            for (int w = 0; w < 2; w++) q.parallel_for(range<1>(N), kernel).wait();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < 10; r++) q.parallel_for(range<1>(N), kernel).wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count() / 10);
        }
        std::printf("  %-30s %7.4f ms\n", nm, best);
        return best;
    };
    const char *tn[6] = {"sycl::tanh (reference)", "exp2 (e-1)/(e+1)", "exp (e-1)/(e+1)",
                         "expm1 m/(m+2)", "exp2 1-2/(e+1)", "clamp>9.011 + sycl::tanh"};
    for (int sat = 0; sat < 2; sat++) {
        fill(sat);
        std::printf("\nN=%zu  distribution: %s\n", N, sat ? "|x| up to 60 (saturated)" : "|x| up to 0.7");
        auto k0 = [=](id<1> i) { y[i] = cand<0>(x[i]); };
        auto k1 = [=](id<1> i) { y[i] = cand<1>(x[i]); };
        auto k2 = [=](id<1> i) { y[i] = cand<2>(x[i]); };
        auto k3 = [=](id<1> i) { y[i] = cand<3>(x[i]); };
        auto k4 = [=](id<1> i) { y[i] = cand<4>(x[i]); };
        auto k5 = [=](id<1> i) { y[i] = cand<5>(x[i]); };
        bench(tn[0], k0);
        bench(tn[1], k1);
        bench(tn[2], k2);
        bench(tn[3], k3);
        bench(tn[4], k4);
        bench(tn[5], k5);
    }
    return 0;
}
