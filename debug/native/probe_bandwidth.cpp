/*
 * Real DRAM read/write/copy ceilings at the buffer sizes the pipeline uses.
 * This supersedes the old "788 GB/s streaming" figure in the README, which
 * came from a buffer that fit in the 18 MB L2 and was re-read in a loop.
 *
 *   ./build.sh && ./bin/probe_bandwidth
 */
// Measure this GPU's real read / write / mixed bandwidth ceilings at the exact
// sizes the pipeline uses, so the "floor" column in the README is grounded.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
using namespace sycl;

int main() {
    queue q;
    struct { const char *nm; size_t bytes; } cases[] = {
        {"12.6 MB (int8 out)",  12600000},
        {"25.2 MB (bf16 in)",   25200000},
        {"50.3 MB (int32 acc)", 50300000},
        {"100 MB",             100000000},
        {"400 MB",             400000000},
    };
    std::printf("%-22s %12s %12s %12s %12s\n", "size", "read", "write", "copy r+w", "r+w GB");
    for (auto &c : cases) {
        size_t n = c.bytes;
        size_t nf = n / 4;
        float *a = malloc_shared<float>(nf, q);
        float *b = malloc_shared<float>(nf, q);
        float *d = malloc_shared<float>(nf, q);
        auto bench = [&](auto fn, double bytes) {
            for (int i = 0; i < 3; ++i) fn();
            auto t0 = std::chrono::high_resolution_clock::now();
            int it = n > 1e8 ? 5 : 20;
            for (int i = 0; i < it; ++i) fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / it;
            return bytes / 1e9 / (ms * 1e-3);
        };
        double rd = bench([&] {
            q.submit([&](handler &h) { h.parallel_for(range<1>(nf), [=](size_t i) {
                d[i] = a[i] * 1.000001f; }); }).wait();
        }, (double)n);
        double wr = bench([&] {
            q.submit([&](handler &h) { h.parallel_for(range<1>(nf), [=](size_t i) {
                d[i] = (float)(i & 7); }); }).wait();
        }, (double)n);
        double cp = bench([&] {
            q.submit([&](handler &h) { h.parallel_for(range<1>(nf), [=](size_t i) {
                d[i] = a[i] + b[i]; }); }).wait();
        }, (double)2 * n);
        std::printf("%-22s %9.0f GB/s %9.0f GB/s %9.0f GB/s %12.0f\n",
                    c.nm, rd, wr, cp, rd + wr);
        free(a, q); free(b, q); free(d, q);
    }
    return 0;
}