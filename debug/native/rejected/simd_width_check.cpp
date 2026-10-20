/* NEGATIVE RESULT -- kept as evidence.
 * Is vec<float,16> really 512-bit, or silently 256-bit?
 * RESULT: ~10 T float-ops/s, i.e. near hardware peak. SIMD width is fine.
 */
// Is vec<float,16> really 512-bit here, or does it silently become 256-bit?
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
using namespace sycl;

template <int W>
static void kern(queue &q, float *out, size_t n, int iters) {
    using V = vec<float, W>;
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(n), [=](id<1> i) {
            V a((float)i[0] + 1.f), b((float)i[0] + 2.f);
            V r = a;
            for (int k = 0; k < iters; k++) r = r * b + a;
            float s = 0;
            for (int t = 0; t < W; t++) s += r[t];
            out[i[0]] = s;
        });
    }).wait();
}

int main() {
    queue q;
    size_t n = 1 << 20;
    float *o = malloc_shared<float>(n, q);
    const int iters = 64;
    auto run = [&](auto fn, const char *nm, int w) {
        for (int i = 0; i < 3; i++) fn();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10;
        // 2 ops (mul,add) per iteration per lane
        double ops = (double)n * iters * 2 * w;
        std::printf("%-18s %.4f ms  %.1f G float-ops/s  (%.2f G vec-ops/s)\n", nm, ms,
                    ops / (ms * 1e-3) / 1e9, (double)n * iters * 2 / (ms * 1e-3) / 1e9);
        (void)w;
    };
    run([&] { kern<4>(q, o, n, iters); }, "vec<float,4>", 4);
    run([&] { kern<8>(q, o, n, iters); }, "vec<float,8>", 8);
    run([&] { kern<16>(q, o, n, iters); }, "vec<float,16>", 16);
    return 0;
}