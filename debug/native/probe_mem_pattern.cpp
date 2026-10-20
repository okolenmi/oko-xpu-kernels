/*
 * Memory pattern probe for the ConvRot quantize kernel.
 *
 * The production kernel maps one work-item to one 256-element ConvRot group:
 * the 16 registers of a group live in one lane, so the lane reads 512 B
 * contiguous and writes 256 B contiguous, but the 16 lanes of the sub-group
 * are 512 B apart.  At any single load instruction the warp therefore
 * touches 16 addresses 512 B apart, and each 32 B request sits alone in a
 * 128 B line.
 *
 * The alternative layout maps one *group* to a whole 16-lane sub-group:
 * lane L holds register L of the group, so the warp's single load covers
 * 512 B fully contiguous and the store covers 256 B fully contiguous.  The
 * four Kronecker axes then need two cross-lane butterflies (xor masks 1,2
 * for axis 2 and 4,8 for axis 3) instead of register shuffles.
 *
 * This probe measures the two access patterns with the transform removed, so
 * it answers "is the layout the problem" without the transform confusing the
 * answer.  A third variant keeps the production layout but stores with 32 B
 * writes, to separate the read pattern from the write pattern.
 *
 *   ./build.sh && ./bin/probe_mem_pattern 3952 3072
 */
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
using namespace sycl;

constexpr int GS = 256, NREG = 16, SUBG = 16;

static double bench(queue &q, const std::function<void()> &fn, int iters = 30) {
    for (int i = 0; i < 5; i++) fn();
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) fn();
    q.wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main(int argc, char **argv) {
    const size_t M = argc > 1 ? atoi(argv[1]) : 3952;
    const size_t K = argc > 2 ? atoi(argv[2]) : 3072;
    const size_t ng = K / GS;

    queue q;
    std::cout << "device      : " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    uint16_t *x = malloc_shared<uint16_t>(M * K, q);
    int8_t *o = malloc_shared<int8_t>(M * K, q);
    for (size_t i = 0; i < M * K; i++) x[i] = 0x3c3c;

    const double bytes = (double)M * K * 3.0;  // 2 B in + 1 B out per element

    // ---- A: production layout, lane owns a whole group ---------------------
    auto runA = [&] {
        q.submit([&](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(M * SUBG), range<1>(SUBG)),
                           [=](nd_item<1> it) {
                               const size_t row = it.get_global_id(0) >> 4;
                               const size_t lane = it.get_local_id(0) & 15;
                               const uint16_t *p = x + row * K + lane * GS;
                               int8_t *d = o + row * K + lane * GS;
                               if (lane >= ng) return;
                               for (int w = 0; w < NREG; w++) {
                                   const vec<uint32_t, 8> u =
                                       reinterpret_cast<const vec<uint32_t, 8> *>(p + w * 16)[0];
                                   uint32_t b[4] = {0, 0, 0, 0};
                                   for (int t = 0; t < 8; t++)
                                       b[w >> 1] |= (u[t] & 0xFFu) << (8 * (t & 3));
                                   std::memcpy(d + w * 16, b, 16);
                               }
                           });
        }).wait();
    };

    // ---- B: sub-group owns a group, lane owns one register ----------------
    // One work-item per register; the 16 lanes of a group are consecutive
    // work-items, so a warp's single load covers 512 B contiguous.
    auto runB = [&] {
        q.submit([&](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(M * ng * SUBG), range<1>(SUBG)),
                           [=](nd_item<1> it) {
                               const size_t gid = it.get_global_id(0);
                               const size_t lane = it.get_local_id(0) & 15;
                               const size_t grp = gid / SUBG;      // group index
                               const size_t row = grp / ng;        // row
                               const size_t g = grp - row * ng;    // group in row
                               const uint16_t *p = x + row * K + g * GS + lane * 16;
                               int8_t *d = o + row * K + g * GS + lane * 16;
                               const vec<uint32_t, 8> u =
                                   reinterpret_cast<const vec<uint32_t, 8> *>(p)[0];
                               uint32_t b[4] = {0, 0, 0, 0};
                               for (int t = 0; t < 8; t++)
                                   b[t >> 1] |= (u[t] & 0xFFu) << (8 * (t & 3));
                               std::memcpy(d, b, 16);
                           });
        }).wait();
    };

    // ---- C: production layout, but one 32 B store per lane ----------------
    auto runC = [&] {
        q.submit([&](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(M * SUBG), range<1>(SUBG)),
                           [=](nd_item<1> it) {
                               const size_t row = it.get_global_id(0) >> 4;
                               const size_t lane = it.get_local_id(0) & 15;
                               const uint16_t *p = x + row * K + lane * GS;
                               int8_t *d = o + row * K + lane * GS;
                               if (lane >= ng) return;
                               for (int w = 0; w < NREG; w += 2) {
                                   const vec<uint32_t, 8> u0 =
                                       reinterpret_cast<const vec<uint32_t, 8> *>(p + w * 16)[0];
                                   const vec<uint32_t, 8> u1 =
                                       reinterpret_cast<const vec<uint32_t, 8> *>(p + (w + 1) * 16)[0];
                                   uint32_t b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                                   for (int t = 0; t < 8; t++) {
                                       b[t >> 1] |= (u0[t] & 0xFFu) << (8 * (t & 3));
                                       b[4 + (t >> 1)] |= (u1[t] & 0xFFu) << (8 * (t & 3));
                                   }
                                   std::memcpy(d + w * 16, b, 32);
                               }
                           });
        }).wait();
    };

    const double a = bench(q, runA), b = bench(q, runB), c = bench(q, runC);
    auto show = [&](const char *n, double ms) {
        std::printf("%-52s %8.4f ms  %7.1f GB/s\n", n, ms, bytes / ms * 1e-6);
    };
    std::printf("\nM=%zu K=%zu  (%.1f MB in + %.1f MB out)\n", M, K,
                M * K * 2.0 / 1048576.0, M * K / 1048576.0);
    show("A production layout  (lane = 256-elem group)", a);
    show("B sub-group = group  (lane = one 16-elem register)", b);
    show("C production layout, 32 B stores", c);
    std::printf("\nB vs A: %.2fx   C vs A: %.2fx\n", a / b, a / c);
    return 0;
}
