/*
 * Candidate replacement for the ConvRot rotate+quantize kernel.
 *
 * The production kernel maps one work-item to one 256-element ConvRot group:
 * all 16 registers of the group live in one lane, so the lane reads 512 B
 * contiguous and writes 256 B contiguous, but the 16 lanes of a sub-group are
 * 512 B apart.  At a single load instruction the warp therefore touches 16
 * addresses 512 B apart and each 32 B request sits alone in a 128 B line:
 * probe_mem_pattern.cpp measures 195 GB/s for that pattern against 454 GB/s
 * for a coalesced one, a 2.33x gap.
 *
 * This candidate maps a ConvRot group to a whole sub-group instead: lane L
 * holds register L of the group.  Then
 *   - the warp's single load covers 512 B fully contiguous,
 *   - the warp's single store covers 256 B fully contiguous,
 *   - each lane holds exactly ONE register, so the transformed value can be
 *     kept across the row-scale reduction -- one read, one write, one
 *     transform, no staging buffer.
 * The two cross-register Kronecker axes become cross-lane butterflies:
 * axis 2 (registers 4g..4g+3) is xor masks 1 and 2, axis 3 (registers
 * c, c+4, c+8, c+12) is xor masks 4 and 8.  The in-register axes 0 and 1 are
 * unchanged.
 *
 * A 4-point Hadamard butterfly over lanes {L, L^m1, L^m2, L^m1^m2} with
 * m1^m2 the third partner decomposes as
 *     A = v + shuffle_xor(v, m1)
 *     B = v - shuffle_xor(v, m1)
 *     out = (A + shuffle_xor(B, m2)) * 0.5
 * which reproduces bfly4() exactly (checked bit-for-bit against the
 * production kernel below).
 *
 * The row scale needs a max over the whole row, and a row spans n_groups
 * groups, so the work-group is sized to cover a whole row: n_groups sub-groups
 * of 16 lanes.  K=3072 -> 12 sub-groups (192 threads), K=8192 -> 32
 * sub-groups (512 threads).  The sub-group maxima go through local memory,
 * one barrier, and every lane then reduces them itself.
 *
 *   ./build.sh && ./bin/probe_quant_warp 3952 3072
 */
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
using namespace sycl;

constexpr int GS = 256, NREG = 16, SUBG = 16;

static const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;
using BF = sycl::ext::oneapi::bfloat16;
using BF16 = vec<BF, 16>;

static inline F16 h4_col(int c, bool d1) {
    F16 v;
    for (int t = 0; t < 16; t++) v[t] = 0.5f * H4[d1 ? ((t >> 2) & 3) : (t & 3)][c];
    return v;
}
static inline F16 load_reg(const uint16_t *p) {
    const vec<uint32_t, 8> u = reinterpret_cast<const vec<uint32_t, 8> *>(p)[0];
    U16 w;
    for (int j = 0; j < 8; j++) {
        w[2 * j] = u[j] << 16;
        w[2 * j + 1] = u[j] & 0xFFFF0000u;
    }
    return sycl::bit_cast<F16>(w);
}
static inline void store_reg(int8_t *p, F16 v, float inv) {
    const F16 r = sycl::min(sycl::max(sycl::rint(v * inv), F16(-127.f)), F16(127.f));
    uint32_t b[4] = {0, 0, 0, 0};
    for (int t = 0; t < 16; t++)
        b[t >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(
                         static_cast<int8_t>(static_cast<int32_t>(r[t]))))
                     << (8 * (t & 3));
    std::memcpy(p, b, 16);
}
static inline void bfly4(F16 &a0, F16 &a1, F16 &a2, F16 &a3) {
    const F16 p = a0 + a1, m = a0 - a1, q = a2 + a3, n = a2 - a3;
    a0 = (p + n) * 0.5f;
    a1 = (p - n) * 0.5f;
    a2 = (m + q) * 0.5f;
    a3 = (q - m) * 0.5f;
}
static inline F16 inreg(F16 v, const F16 &c0, const F16 &c1, const F16 &c2, const F16 &c3,
                        const F16 &d0, const F16 &d1, const F16 &d2, const F16 &d3) {
    F16 t = c0 * v.swizzle<0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12>();
    t = t + c1 * v.swizzle<1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13>();
    t = t + c2 * v.swizzle<2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14>();
    t = t + c3 * v.swizzle<3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15>();
    F16 u = d0 * t.swizzle<0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>();
    u = u + d1 * t.swizzle<4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7>();
    u = u + d2 * t.swizzle<8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11>();
    u = u + d3 * t.swizzle<12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15>();
    return u;
}

// ---------------------------------------------------------------- production
// Baseline: the production kernel, written with a FLAT local array for the
// one-group case.  Passing a local array by pointer into a lambda makes the
// compiler demote it to scratch memory, which costs ~2x -- that is a probe
// artefact, not a property of the library kernel, so the baseline has to be
// the flat form to be a fair reference.
template <int NGRP>
static void prod_body(nd_item<1> it, const uint16_t *x, int8_t *qout, float *scale,
                      size_t M, size_t K, size_t n_groups, const F16 &c0, const F16 &c1,
                      const F16 &c2, const F16 &c3, const F16 &d0, const F16 &d1,
                      const F16 &d2, const F16 &d3) {
    const size_t row = it.get_global_id(0) >> 4;
    const size_t lane = it.get_local_id(0) & 15;
    const uint16_t *xr = x + row * K;
    F16 amaxv(0.f);
    F16 W[NGRP == 1 ? 1 : NGRP][NREG];
    if (NGRP == 1) {
        if (lane < n_groups) {
            const uint16_t *p = xr + lane * GS;
            for (int b = 0; b < NREG; b += 4) {
                for (int j = 0; j < 4; j++) W[0][b + j] = load_reg(p + (b + j) * 16);
                for (int j = 0; j < 4; j++)
                    W[0][b + j] = inreg(W[0][b + j], c0, c1, c2, c3, d0, d1, d2, d3);
                bfly4(W[0][b], W[0][b + 1], W[0][b + 2], W[0][b + 3]);
            }
            for (int c = 0; c < 4; c++)
                bfly4(W[0][c], W[0][c + 4], W[0][c + 8], W[0][c + 12]);
            for (int w = 0; w < NREG; ++w) amaxv = sycl::max(amaxv, sycl::fabs(W[0][w]));
        }
    } else {
        for (int g = 0; g < NGRP; ++g) {
            const size_t grp = g * SUBG + lane;
            if (grp >= n_groups) continue;
            const uint16_t *p = xr + grp * GS;
            for (int b = 0; b < NREG; b += 4) {
                for (int j = 0; j < 4; j++) W[g][b + j] = load_reg(p + (b + j) * 16);
                for (int j = 0; j < 4; j++)
                    W[g][b + j] = inreg(W[g][b + j], c0, c1, c2, c3, d0, d1, d2, d3);
                bfly4(W[g][b], W[g][b + 1], W[g][b + 2], W[g][b + 3]);
            }
            for (int c = 0; c < 4; c++)
                bfly4(W[g][c], W[g][c + 4], W[g][c + 8], W[g][c + 12]);
            for (int w = 0; w < NREG; ++w) amaxv = sycl::max(amaxv, sycl::fabs(W[g][w]));
        }
    }
    F16 r0 = sycl::max(amaxv, amaxv.swizzle<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>());
    F16 r1 = sycl::max(r0, r0.swizzle<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>());
    F16 r2 = sycl::max(r1, r1.swizzle<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>());
    F16 r3 = sycl::max(r2, r2.swizzle<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>());
    const auto sg = it.get_sub_group();
    float v = r3[0];
    v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 8)));
    v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 4)));
    v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 2)));
    v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 1)));
    const float s = sycl::fmax(v / 127.0f, 1e-30f);
    if (lane == 0) scale[row] = s;
    const float inv = 1.0f / s;
    if (NGRP == 1) {
        if (lane < n_groups) {
            int8_t *p = qout + row * K + lane * GS;
            for (int w = 0; w < NREG; ++w) store_reg(p + w * 16, W[0][w], inv);
        }
    } else {
        for (int g = 0; g < NGRP; ++g) {
            const size_t grp = g * SUBG + lane;
            if (grp >= n_groups) continue;
            int8_t *p = qout + row * K + grp * GS;
            for (int w = 0; w < NREG; ++w) store_reg(p + w * 16, W[g][w], inv);
        }
    }
}

template <int NGRP>
static void prod_kernel_impl(queue &q, const uint16_t *x, int8_t *qout, float *scale,
                              size_t M, size_t K) {
    const size_t ng = K / GS;
    const F16 c0 = h4_col(0, 0), c1 = h4_col(1, 0), c2 = h4_col(2, 0), c3 = h4_col(3, 0);
    const F16 d0 = h4_col(0, 1), d1 = h4_col(1, 1), d2 = h4_col(2, 1), d3 = h4_col(3, 1);
    auto ev = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M * SUBG), range<1>(SUBG)), [=](nd_item<1> it) {
            prod_body<NGRP>(it, x, qout, scale, M, K, ng, c0, c1, c2, c3, d0, d1, d2, d3);
        });
    });
    ev.wait();
}

// The two instantiations must be separate kernels: the NGRP==2 variant needs
// 32 live vector registers, and if the compiler sees both in one kernel the
// register allocator sizes the whole kernel for the worse case.
static void prod_kernel(queue &q, const uint16_t *x, int8_t *qout, float *scale, size_t M,
                        size_t K) {
    if (K / GS <= SUBG)
        prod_kernel_impl<1>(q, x, qout, scale, M, K);
    else
        prod_kernel_impl<2>(q, x, qout, scale, M, K);
}

// ------------------------------------------------------------------ candidate
// Cross-lane 4-point Hadamard butterfly: lanes {L, L^m1, L^m2, L^m1^m2}.
static inline F16 bfly_xor(sub_group sg, F16 v, int m1, int m2) {
    const uint32_t me = (uint32_t)sg.get_local_linear_id();
    const F16 p = sycl::select_from_group(sg, v, me ^ (uint32_t)m1);
    const F16 A = v + p;
    const F16 B = v - p;
    const F16 Bp = sycl::select_from_group(sg, B, me ^ (uint32_t)m2);
    return (A + Bp) * 0.5f;
}

static void warp_kernel(queue &q, const uint16_t *x, int8_t *qout, float *scale, size_t M,
                        size_t K) {
    const size_t ng = K / GS;          // groups per row
    const size_t wg = ng * SUBG;       // threads per work-group == one row
    const F16 c0 = h4_col(0, 0), c1 = h4_col(1, 0), c2 = h4_col(2, 0), c3 = h4_col(3, 0);
    const F16 d0 = h4_col(0, 1), d1 = h4_col(1, 1), d2 = h4_col(2, 1), d3 = h4_col(3, 1);
    auto ev = q.submit([&](handler &h) {
        local_accessor<float, 1> lmax(range<1>(ng), h);
        h.parallel_for(nd_range<1>(range<1>(M * wg), range<1>(wg)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) / wg;
            const size_t lid = it.get_local_id(0);
            const size_t sgid = lid / SUBG;   // which group of the row
            const size_t lane = lid % SUBG;   // which register of the group
            const uint16_t *p = x + row * K + sgid * GS + lane * 16;

            F16 v = load_reg(p);
            v = inreg(v, c0, c1, c2, c3, d0, d1, d2, d3);
            const auto sg = it.get_sub_group();
            v = bfly_xor(sg, v, 1, 2);   // axis 2
            v = bfly_xor(sg, v, 4, 8);   // axis 3

            F16 a = sycl::fabs(v);
            F16 r0 = sycl::max(a, a.swizzle<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>());
            F16 r1 = sycl::max(r0, r0.swizzle<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>());
            F16 r2 = sycl::max(r1, r1.swizzle<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>());
            F16 r3 = sycl::max(r2, r2.swizzle<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>());
            const uint32_t me = (uint32_t)sg.get_local_linear_id();
            float vm = r3[0];
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 8u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 4u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 2u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 1u));
            if (lane == 0) lmax[sgid] = vm;

            sycl::group_barrier(it.get_group());
            float rowmax = 0.f;
            for (size_t g = 0; g < ng; ++g) rowmax = sycl::fmax(rowmax, lmax[g]);
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (lid == 0) scale[row] = s;
            store_reg(qout + row * K + sgid * GS + lane * 16, v, 1.0f / s);
        });
    });
    ev.wait();
}

template <typename T>
static T *alloc_buf(size_t n, queue &q, bool shared) {
    return shared ? malloc_shared<T>(n, q) : malloc_device<T>(n, q);
}

int main(int argc, char **argv) {
    const size_t M = argc > 1 ? atoi(argv[1]) : 3952;
    const size_t K = argc > 2 ? atoi(argv[2]) : 3072;
    queue q;
    std::printf("device      : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    {
        // probe the hardware sub-group size from inside a kernel
        int *sgsz = malloc_shared<int>(1, q);
        *sgsz = 0;
        q.submit([&](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(32), range<1>(32)),
                           [=](nd_item<1> it) {
                               if (it.get_local_id(0) == 0)
                                   *sgsz = (int)it.get_sub_group().get_local_range().size();
                           });
        }).wait();
        std::printf("sub-group   : %d\n", *sgsz);
    }

    // argv[3]: "shared" (what the other probes use) or "device"
    const bool use_shared = (argc > 3 && std::string(argv[3]) == "shared");
    uint16_t *x = alloc_buf<uint16_t>(M * K, q, use_shared);
    int8_t *qa = alloc_buf<int8_t>(M * K, q, use_shared);
    int8_t *qb = alloc_buf<int8_t>(M * K, q, use_shared);
    float *sa = alloc_buf<float>(M, q, use_shared);
    float *sb = alloc_buf<float>(M, q, use_shared);
    std::printf("memory      : %s\n", use_shared ? "shared USM" : "device USM");
    std::vector<uint16_t> hx(M * K);
    // argv[4]: data mode.  0 = constant 1.0 (what probe_rotate_ablation uses),
    // 1 = skewed, 2 = gaussian.  The choice matters: the transform's partial
    // sums cancel, so a skewed distribution produces denormal intermediates
    // and denormal arithmetic is not full speed on this hardware.
    const int dmode = argc > 4 ? atoi(argv[4]) : 2;
    {
        std::vector<float> hf(M * K);
        srand(1234);
        for (size_t i = 0; i < M * K; i++) {
            float u = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
            if (dmode == 0) hf[i] = 1.0f;
            else if (dmode == 1) hf[i] = u * u * u;
            else {
                float g1 = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
                float g2 = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
                hf[i] = g1 * g2;   // product of two uniforms ~ bell shaped
            }
        }
        for (size_t i = 0; i < M * K; i++) {
            float v = hf[i];
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            uint32_t sign = bits >> 31, exp = (bits >> 23) & 0xFF, man = bits & 0x7FFFFF;
            if (exp == 0) { hx[i] = (uint16_t)(sign << 15); continue; }
            if (exp == 0xFF) { hx[i] = (uint16_t)((sign << 15) | 0x7F80); continue; }
            uint32_t r = man + 0x4000 + ((man >> 7) & 1);
            if (r & 0x80000) { r = 0; exp++; }
            hx[i] = (uint16_t)((sign << 15) | (exp << 7) | ((r >> 8) & 0x7F));
        }
    }
    q.memcpy(x, hx.data(), M * K * 2).wait();

    prod_kernel(q, x, qa, sa, M, K);
    warp_kernel(q, x, qb, sb, M, K);

    std::vector<int8_t> ha(M * K), hb(M * K);
    std::vector<float> hsa(M), hsb(M);
    q.memcpy(ha.data(), qa, M * K).wait();
    q.memcpy(hb.data(), qb, M * K).wait();
    q.memcpy(hsa.data(), sa, M * sizeof(float)).wait();
    q.memcpy(hsb.data(), sb, M * sizeof(float)).wait();

    size_t diff = 0, sdiff = 0;
    float worst = 0.f;
    for (size_t i = 0; i < M * K; i++) {
        if (ha[i] != hb[i]) {
            if (diff < 8) std::printf("  q[%zu] prod=%d warp=%d\n", i, ha[i], hb[i]);
            diff++;
        }
    }
    for (size_t i = 0; i < M; i++) {
        float d = std::fabs(hsa[i] - hsb[i]) / (hsa[i] != 0 ? hsa[i] : 1.f);
        worst = std::max(worst, d);
        if (hsa[i] != hsb[i]) sdiff++;
    }
    std::printf("\nM=%zu K=%zu  q mismatches=%zu / %zu   scale mismatches=%zu (worst rel %.3g)\n",
                M, K, diff, M * K, sdiff, worst);

    auto bench = [&](const char *name, auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < 5; i++) fn();
            q.wait();
            auto t0 = std::chrono::steady_clock::now();
            const int iters = 20;
            for (int i = 0; i < iters; i++) fn();
            q.wait();
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
        }
        std::printf("%-16s %8.4f ms  %7.1f GB/s\n", name, best, M * K * 3.0 / best * 1e-6);
        return best;
    };
    std::printf("\n");
    double a = bench("production", [&] { prod_kernel(q, x, qa, sa, M, K); });
    double b = bench("warp", [&] { warp_kernel(q, x, qb, sb, M, K); });
    std::printf("\nspeedup: %.2fx\n", a / b);
    return 0;
}
