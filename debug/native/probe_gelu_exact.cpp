/*
 * Can the erf-GELU inside SDXL's GEGLU be reproduced bit-for-bit?
 *
 * SDXL's FFN is a GEGLU:  x, gate = proj(x).chunk(2, -1);  out = x * gelu(gate)
 * (comfy/ldm/modules/attention.py:166, and the training project's
 * nodes/model/attention.py:146).  The library already fuses MossImage's
 * equivalent (the "situ" activation) into the FFN, and that fusion is
 * bit-exact -- the tanh it needs is reproduced exactly, because
 * probe_tanh_exact.cpp shows sycl::tanh and the expm1 form agree on every
 * bf16 input.
 *
 * The erf-GELU is the open question: torch's F.gelu(approximate="none") on
 * this backend goes through oneDNN's erf, and a SYCL kernel's sycl::erf is
 * SVML's erf.  Whether the two agree *bit for bit* is not a thing to assume;
 * it is a thing to measure, over the 65536 bf16 values there are.
 *
 * The reference is torch's own output, dumped by
 *   debug/probe_gelu_torch.py
 * as the raw bf16 bit patterns.  The comparison is on the *bf16-rounded*
 * result, because that is what the fused kernel produces: the dequantized
 * value is rounded to bf16 before the activation (the same round the bf16
 * store/reload used to impose), and the activation's result is rounded to
 * bf16 again before the multiply.
 *
 *   ./build.sh && ./bin/probe_gelu_exact build/gelu_torch_erf.bin
 *   ./bin/probe_gelu_exact build/gelu_torch_tanh.bin
 */
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;

static inline float bf_round(float x) { return float(BF(x)); }

// The candidate GELUs.  Each returns the fp32 value; the comparison rounds it
// to bf16, which is what the kernel stores.
template <int C>
static inline float cand(float x) {
    if constexpr (C == 0) {
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5f * x * (1.0f + sycl::erf(x * 0.7071067811865476f));
    } else if constexpr (C == 1) {
        // the same, with the multiplication order changed
        return x * 0.5f * (1.0f + sycl::erf(x * 0.7071067811865476f));
    } else if constexpr (C == 2) {
        // 0.5 * x * (1 + erf(x / sqrt(2))), with a division
        return 0.5f * x * (1.0f + sycl::erf(x / 1.4142135623730951f));
    } else if constexpr (C == 3) {
        // the erfc form: 0.5 * x * erfc(-x / sqrt(2))
        return 0.5f * x * sycl::erfc(-x * 0.7071067811865476f);
    } else if constexpr (C == 4) {
        // the tanh approximation (what the library already ships for
        // MossImage's gelu_tanh)
        const float x3 = x * x * x;
        return 0.5f * x * (1.0f + sycl::tanh(0.7978845608028654f *
                                             (x + 0.044715f * x3)));
    } else {
        // 0.5 * x * (1 + erf(x/sqrt(2))), with the product done in double
        return float(0.5 * double(x) *
                     (1.0 + sycl::erf(double(x) * 0.7071067811865476)));
    }
}

int main(int argc, char **argv) {
    queue q;
    constexpr int N = 65536;
    constexpr int NC = 6;

    std::vector<uint16_t> ref;
    if (argc > 1) {
        std::ifstream f(argv[1], std::ios::binary);
        if (!f) { std::printf("cannot open %s\n", argv[1]); return 1; }
        ref.resize(N);
        f.read((char *)ref.data(), N * 2);
        std::printf("reference: %s\n", argv[1]);
    } else {
        std::printf("no reference given; this only prints the candidates' own summary\n");
    }

    uint16_t *in = malloc_shared<uint16_t>(N, q);
    float *out = malloc_shared<float>(N * NC, q);
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

    const char *names[NC] = {
        "0.5*x*(1+erf(x*1/sqrt2))", "x*0.5*(1+erf(x*1/sqrt2))",
        "0.5*x*(1+erf(x/sqrt2))",   "0.5*x*erfc(-x*1/sqrt2)",
        "the tanh approximation",   "the same in double",
    };

    for (int c = 0; c < NC; c++) {
        long bad_bf = 0, bad_fp = 0, bad_nan = 0;
        double maxrel = 0;
        int worst = -1;
        for (int i = 0; i < N; i++) {
            const float v = out[c * N + i];
            if (std::isnan(v)) { bad_nan++; continue; }
            float r = 0.f;
            if (!ref.empty()) {
                r = float(sycl::bit_cast<BF>(ref[i]));
                const float rb = bf_round(r), vb = bf_round(v);
                if (std::memcmp(&rb, &vb, 4) != 0) {
                    bad_bf++;
                    if (worst < 0) worst = i;
                }
            }
            if (r != 0 && std::isfinite(r))
                maxrel = std::max(maxrel, std::fabs(double(v) - r) / std::fabs(double(r)));
        }
        (void)bad_fp;
        std::printf("  %-28s bf16 mismatch %6ld / 65536   NaN %ld   max rel %.2e",
                    names[c], bad_bf, bad_nan, maxrel);
        if (worst >= 0) {
            const float r = float(sycl::bit_cast<BF>(ref[worst]));
            std::printf("   first mismatch at bf16 0x%04x (ref %g, got %g)",
                        ref[worst], r, out[c * N + worst]);
        }
        std::printf("\n");
    }
    return 0;
}
