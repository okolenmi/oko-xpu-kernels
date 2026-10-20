// Fused INT8 dequant epilogue: acc[M,N] (int32) -> out[M,N] (bf16)
//   out[m,n] = bf16(acc[m,n] * x_scale[m] * w_scale[n] + bias[n])
//
// comfy-kitchen's eager backend does this with a chunked loop that materialises
// float32 intermediates: `chunk.float() * (w_scale * x_scale)` per chunk, then
// bias, then `.to(dtype)`. That costs 1.31 ms at M=4096,N=3072 for 75 MB of
// unavoidable traffic (read int32, write bf16) -- ~7x off bandwidth.
//
// This does it in one pass: one read of int32, one write of bf16, everything
// else in registers. Fully memory bound.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sycl;

using BF = sycl::ext::oneapi::bfloat16;
using BF16 = vec<BF, 16>;
using F16 = vec<float, 16>;
using I32 = vec<int32_t, 16>;

static float f32_from_bf16(BF b) {
    uint16_t u;
    std::memcpy(&u, &b, 2);
    uint32_t w = static_cast<uint32_t>(u) << 16;
    float f;
    std::memcpy(&f, &w, 4);
    return f;
}

int main(int argc, char **argv) {
    const size_t M = argc > 1 ? strtoul(argv[1], nullptr, 10) : 4096;
    const size_t N = argc > 2 ? strtoul(argv[2], nullptr, 10) : 3072;
    const int with_bias = argc > 3 ? atoi(argv[3]) : 1;

    queue q;
    std::printf("%s  M=%zu N=%zu bias=%d\n",
                q.get_device().get_info<info::device::name>().c_str(), M, N, with_bias);

    std::vector<int32_t> ha(M * N);
    std::vector<float> hs(M), hw(N), hb(N);
    uint64_t s = 0x123456789ABCDEFull;
    for (size_t i = 0; i < ha.size(); i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        ha[i] = static_cast<int32_t>((s >> 40) % 200001) - 100000;
    }
    for (size_t i = 0; i < M; i++) hs[i] = 0.001f + (float)(i % 97) * 1e-5f;
    for (size_t i = 0; i < N; i++) hw[i] = 0.002f + (float)(i % 89) * 1e-5f;
    for (size_t i = 0; i < N; i++) hb[i] = (float)((i % 17) - 8) * 0.01f;

    int32_t *acc = malloc_shared<int32_t>(M * N, q);
    BF *out = malloc_shared<BF>(M * N, q);
    float *xs = malloc_shared<float>(M, q);
    float *ws = malloc_shared<float>(N, q);
    float *bs = malloc_shared<float>(N, q);
    for (size_t i = 0; i < ha.size(); i++) acc[i] = ha[i];
    for (size_t i = 0; i < M; i++) xs[i] = hs[i];
    for (size_t i = 0; i < N; i++) { ws[i] = hw[i]; bs[i] = hb[i]; }

    auto run = [&] {
        q.submit([&](handler &h) {
            const size_t nvec = N / 16;
            h.parallel_for(range<2>(M, nvec), [=](id<2> idx) {
                const size_t m = idx[0], nv = idx[1];
                const I32 a = reinterpret_cast<const I32 *>(acc + m * N)[nv];
                const F16 w = reinterpret_cast<const F16 *>(ws)[nv];
                const float xv = xs[m];
                F16 v = F16(a.convert<float>()) * (w * xv);
                if (with_bias) v = v + reinterpret_cast<const F16 *>(bs)[nv];
                BF16 r;
                for (int t = 0; t < 16; t++) r[t] = BF(v[t]);
                reinterpret_cast<BF16 *>(out + m * N)[nv] = r;
            });
        }).wait();
    };
    run();

    // verify on a sample
    double worst = 0.0;
    for (size_t m = 0; m < std::min<size_t>(M, 64); m++)
        for (size_t n = 0; n < N; n++) {
            float ref = (float)ha[m * N + n] * hs[m] * hw[n] + (with_bias ? hb[n] : 0.f);
            float got = f32_from_bf16(out[m * N + n]);
            worst = std::max(worst, (double)std::fabs(ref - got) / std::fmax(1.f, std::fabs(ref)));
        }
    std::printf("max rel error: %.5f  %s\n", worst, worst < 0.01 ? "PASS" : "FAIL");

    for (int i = 0; i < 3; i++) run();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++) run();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
    const double gb = (double)M * N * 6.0 / 1e9;
    std::printf("time: %.4f ms  (%.0f GB/s effective)\n", ms, gb / (ms * 1e-3));

    sycl::free(acc, q); sycl::free(out, q); sycl::free(xs, q);
    sycl::free(ws, q); sycl::free(bs, q);
    return worst < 0.01 ? 0 : 1;
}