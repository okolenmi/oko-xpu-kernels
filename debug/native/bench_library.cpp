/*
 * Calls the production C API with SYCL malloc_shared buffers rather than torch
 * tensors, to separate ".so + codegen" cost from "torch allocator" cost.
 *
 *   icpx -O3 -fsycl -I../../include ../../src/xpuint8.cpp bench_library.cpp -o bench_library
 */
// Call the production C API with SYCL malloc_shared buffers instead of torch's,
// to separate ".so + codegen" cost from "torch allocator" cost.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
extern "C" int xpuint8_init(void);
extern "C" int xpuint8_convrot_quantize(const void *, int8_t *, float *, int64_t, int64_t);
extern "C" int xpuint8_dequant_epilogue(const int32_t *, void *, const float *, const float *,
                                       const float *, int64_t, int64_t);

int main(int argc, char **argv) {
    size_t M = argc > 1 ? strtoul(argv[1], 0, 10) : 4096;
    size_t K = argc > 2 ? strtoul(argv[2], 0, 10) : 3072;
    size_t N = argc > 3 ? strtoul(argv[3], 0, 10) : 3072;
    sycl::queue q;
    xpuint8_init();
    uint16_t *x = sycl::malloc_shared<uint16_t>(M * K, q);
    int8_t *o = sycl::malloc_shared<int8_t>(M * K, q);
    float *sc = sycl::malloc_shared<float>(M, q);
    int32_t *acc = sycl::malloc_shared<int32_t>(M * N, q);
    float *ws = sycl::malloc_shared<float>(N, q);
    float *bs = sycl::malloc_shared<float>(N, q);
    for (size_t i = 0; i < M * K; ++i) x[i] = 0x3F80;
    auto bench = [&](const char *nm, auto fn) {
        double best = 1e9;
        for (int r = 0; r < 3; ++r) {
            for (int i = 0; i < 5; ++i) fn();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 20; ++i) fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count() / 20);
        }
        std::printf("%-42s %7.4f ms\n", nm, best);
    };
    std::printf("M=%zu K=%zu N=%zu  (SYCL malloc_shared buffers)\n", M, K, N);
    bench("convrot_quantize", [&] { xpuint8_convrot_quantize(x, o, sc, M, K); });
    bench("dequant_epilogue", [&] { xpuint8_dequant_epilogue(acc, o, sc, ws, bs, M, N); });
    return 0;
}