/*
 * Attribution ablation for the fused FFN dequant+situ+rotate+quantize kernel
 * (launch_dequant_situ_quant_warp in comfy/lib/xpu_int8/src.cpp).
 *
 * Production shape: M=3973 rows, K=8192 (one row of the FFN intermediate),
 * acc is int32 [M, 2K] = 260.4 MB read, qout int8 [M, K] = 32.5 MB written.
 * The kernel was measured at ~1.376 ms in the model, i.e. ~213 GB/s effective
 * for 292.9 MB of traffic, while a pure copy of the same bytes runs at
 * 320-430 GB/s. This probe asks where the missing time goes: transcendentals,
 * the row-max barrier that gates every store, the Hadamard transform, or the
 * int8 pack.
 *
 * Each MODE removes exactly one thing from a byte-for-byte copy of the
 * production kernel, so the deltas read off directly. MODE 0 must reproduce
 * the production time or the probe is not measuring the same code.
 *
 *   ./build.sh && ./bin/probe_ffn_situ 3973 8192
 *   ./bin/probe_ffn_situ 3974 8192
 */
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <utility>
#include <vector>

using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;
using I32 = vec<int32_t, 16>;

constexpr int GS = 256, NREG = 16, SUBG = 16;
static const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};

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

// production situ, step for step (see src.cpp situ_reg)
static inline F16 situ_reg(F16 h, F16 g) {
    F16 tg = sycl::tanh((g / 4.0f).template convert<BF>().template convert<float>());
    tg = tg.template convert<BF>().template convert<float>();
    F16 sg = 1.0f / (1.0f + sycl::exp(-g));
    sg = sg.template convert<BF>().template convert<float>();
    F16 th = sycl::tanh((h / 25.0f).template convert<BF>().template convert<float>());
    th = th.template convert<BF>().template convert<float>();
    F16 p = tg * sg;
    p = p.template convert<BF>().template convert<float>();
    return (p * th).template convert<BF>().template convert<float>();
}

static inline F16 dequant_reg(const I32 &i, const F16 &w, float xs) {
    return (F16(i.convert<float>()) * (w * xs)).template convert<BF>().template convert<float>();
}

static inline F16 load_dequant_situ_reg(const int32_t *hidden, const int32_t *gate,
                                        const float *hidden_w, const float *gate_w,
                                        float x_scale) {
    const I32 hi = reinterpret_cast<const I32 *>(hidden)[0];
    const I32 gi = reinterpret_cast<const I32 *>(gate)[0];
    const F16 hw = reinterpret_cast<const F16 *>(hidden_w)[0];
    const F16 gw = reinterpret_cast<const F16 *>(gate_w)[0];
    return situ_reg(dequant_reg(hi, hw, x_scale), dequant_reg(gi, gw, x_scale));
}

// situ variants used by the attribution modes below. Each keeps the bf16
// rounding steps of the production chain so only the named op changes.
static inline F16 situ_no_tanh(F16 h, F16 g) {
    F16 sg = 1.0f / (1.0f + sycl::exp(-g));
    sg = sg.template convert<BF>().template convert<float>();
    F16 p = (g / 4.0f) * sg;
    p = p.template convert<BF>().template convert<float>();
    return (p * (h / 25.0f)).template convert<BF>().template convert<float>();
}
static inline F16 situ_no_sigmoid(F16 h, F16 g) {
    F16 tg = sycl::tanh((g / 4.0f).template convert<BF>().template convert<float>());
    tg = tg.template convert<BF>().template convert<float>();
    F16 th = sycl::tanh((h / 25.0f).template convert<BF>().template convert<float>());
    th = th.template convert<BF>().template convert<float>();
    F16 p = tg * (g * 0.25f * g);
    p = p.template convert<BF>().template convert<float>();
    return (p * th).template convert<BF>().template convert<float>();
}
static inline F16 situ_no_div(F16 h, F16 g) {
    F16 tg = sycl::tanh((g / 4.0f).template convert<BF>().template convert<float>());
    tg = tg.template convert<BF>().template convert<float>();
    F16 sg = sycl::exp(-g);
    sg = sg.template convert<BF>().template convert<float>();
    F16 th = sycl::tanh((h / 25.0f).template convert<BF>().template convert<float>());
    th = th.template convert<BF>().template convert<float>();
    F16 p = tg * sg;
    p = p.template convert<BF>().template convert<float>();
    return (p * th).template convert<BF>().template convert<float>();
}

// tanh via expm1, used by MODE 13/14. Identical to sycl::tanh after the bf16
// rounding for every finite bf16 input (probe_tanh_exact, all 65536 encodings),
// but without the data-dependent slow path that costs 0.71 ms at M=3974.
static inline F16 tanh_em1(F16 x) {
    const F16 m = sycl::expm1(2.0f * x);
    return m / (m + 2.0f);
}
static inline F16 situ_fast(F16 h, F16 g) {
    F16 tg = tanh_em1((g / 4.0f).template convert<BF>().template convert<float>());
    tg = tg.template convert<BF>().template convert<float>();
    F16 sg = 1.0f / (1.0f + sycl::exp(-g));
    sg = sg.template convert<BF>().template convert<float>();
    F16 th = tanh_em1((h / 25.0f).template convert<BF>().template convert<float>());
    th = th.template convert<BF>().template convert<float>();
    F16 p = tg * sg;
    p = p.template convert<BF>().template convert<float>();
    return (p * th).template convert<BF>().template convert<float>();
}

static inline F16 quantize_vec(F16 v, float inv) {
    return sycl::min(sycl::max(sycl::rint(v * inv), F16(-127.f)), F16(127.f));
}

static inline void store_reg(int8_t *p, F16 v, float inv) {
    const F16 r = quantize_vec(v, inv);
    uint32_t b[4] = {0, 0, 0, 0};
    for (int t = 0; t < 16; t++)
        b[t >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(
                         static_cast<int8_t>(static_cast<int32_t>(r[t]))))
                     << (8 * (t & 3));
    std::memcpy(p, b, 16);
}

// int32-vector pack (the alternative measured in probe_rotate_ablation)
static inline void store_reg_i32(int8_t *p, F16 v, float inv) {
    const I32 i = quantize_vec(v, inv).convert<int32_t>();
    uint32_t b[4];
    for (int j = 0; j < 4; j++)
        b[j] = (static_cast<uint32_t>(i[4 * j]) & 0xFFu) |
               ((static_cast<uint32_t>(i[4 * j + 1]) & 0xFFu) << 8) |
               ((static_cast<uint32_t>(i[4 * j + 2]) & 0xFFu) << 16) |
               ((static_cast<uint32_t>(i[4 * j + 3]) & 0xFFu) << 24);
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

static inline F16 bfly_xor(sub_group sg, F16 v, uint32_t m1, uint32_t m2) {
    const uint32_t me = static_cast<uint32_t>(sg.get_local_linear_id());
    const F16 p = sycl::select_from_group(sg, v, me ^ m1);
    const F16 A = v + p;
    const F16 B = v - p;
    const F16 Bp = sycl::select_from_group(sg, B, me ^ m2);
    return (A + Bp) * 0.5f;
}

// MODE 0 production
//      1 minus situ (dequant + bf16 round, h*g)
//      2 minus row-max/barrier (sub-group max only, no local mem, no barrier)
//      3 minus int8 store (compute everything, no write)
//      4 minus Hadamard transform (inreg + both lane butterflies)
//      5 loads only (read both halves, no compute, no store)
//      6 int32-vector pack instead of the scalar pack
//      7 minus situ AND minus store (read + transform + max)
//      8 minus situ and minus transform (read + dequant + max + store)
template <int MODE>
static void run(queue &q, const int32_t *acc, int8_t *qout, float *scale,
                const float *x_scale, const float *w_scale, size_t M, size_t K) {
    const size_t n_groups = K / GS;
    const size_t wg = n_groups * SUBG;
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    q.submit([&](handler &h) {
        sycl::local_accessor<float, 1> lmax(range<1>(n_groups), h);
        h.parallel_for(nd_range<1>(range<1>(M * wg), range<1>(wg)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) / wg;
            const size_t lid = it.get_local_id(0);
            const size_t sgid = lid / SUBG;
            const size_t lane = lid % SUBG;
            const size_t off = sgid * GS + lane * 16;
            const int32_t *ar = acc + row * (2 * K);

            F16 v;
            if constexpr (MODE == 5) {
                // read exactly the same bytes, keep the sum alive
                const I32 hi = reinterpret_cast<const I32 *>(ar + off)[0];
                const I32 gi = reinterpret_cast<const I32 *>(ar + K + off)[0];
                float s = 0.f;
                for (int t = 0; t < 16; t++) s += float(hi[t]) + float(gi[t]);
                if (lane == 0) scale[row] = s;
                return;
            } else if constexpr (MODE == 1 || MODE == 7) {
                const I32 hi = reinterpret_cast<const I32 *>(ar + off)[0];
                const I32 gi = reinterpret_cast<const I32 *>(ar + K + off)[0];
                const F16 hw = reinterpret_cast<const F16 *>(w_scale + off)[0];
                const F16 gw = reinterpret_cast<const F16 *>(w_scale + K + off)[0];
                v = dequant_reg(hi, hw, x_scale[row]) * dequant_reg(gi, gw, x_scale[row]);
                v = v.template convert<BF>().template convert<float>();
            } else if constexpr (MODE == 8) {
                const I32 hi = reinterpret_cast<const I32 *>(ar + off)[0];
                const I32 gi = reinterpret_cast<const I32 *>(ar + K + off)[0];
                const F16 hw = reinterpret_cast<const F16 *>(w_scale + off)[0];
                const F16 gw = reinterpret_cast<const F16 *>(w_scale + K + off)[0];
                v = dequant_reg(hi, hw, x_scale[row]) + dequant_reg(gi, gw, x_scale[row]);
                v = v.template convert<BF>().template convert<float>();
            } else if constexpr (MODE >= 10) {
                const I32 hi = reinterpret_cast<const I32 *>(ar + off)[0];
                const I32 gi = reinterpret_cast<const I32 *>(ar + K + off)[0];
                const F16 hw = reinterpret_cast<const F16 *>(w_scale + off)[0];
                const F16 gw = reinterpret_cast<const F16 *>(w_scale + K + off)[0];
                const F16 h = dequant_reg(hi, hw, x_scale[row]);
                const F16 g = dequant_reg(gi, gw, x_scale[row]);
                if constexpr (MODE == 10) v = situ_no_tanh(h, g);
                else if constexpr (MODE == 11) v = situ_no_sigmoid(h, g);
                else if constexpr (MODE == 13) v = situ_fast(h, g);
                else v = situ_no_div(h, g);
            } else {
                v = load_dequant_situ_reg(ar + off, ar + K + off, w_scale + off,
                                          w_scale + K + off, x_scale[row]);
            }

            if constexpr (MODE != 4 && MODE != 8) {
                v = inreg(v, c0, c1, c2, c3, d0, d1, d2, d3);
                const auto sg = it.get_sub_group();
                v = bfly_xor(sg, v, 1u, 2u);
                v = bfly_xor(sg, v, 4u, 8u);
            }

            const F16 a = sycl::fabs(v);
            F16 r0 = sycl::max(a, a.swizzle<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>());
            F16 r1 = sycl::max(r0, r0.swizzle<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>());
            F16 r2 = sycl::max(r1, r1.swizzle<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>());
            F16 r3 = sycl::max(r2, r2.swizzle<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>());
            const auto sg = it.get_sub_group();
            const uint32_t me = static_cast<uint32_t>(sg.get_local_linear_id());
            float vm = r3[0];
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 8u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 4u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 2u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 1u));

            float rowmax;
            if constexpr (MODE == 2) {
                rowmax = vm;  // skip the whole-row dependency
            } else {
                if (lane == 0) lmax[sgid] = vm;
                sycl::group_barrier(it.get_group());
                rowmax = 0.f;
                for (size_t g = 0; g < n_groups; ++g) rowmax = sycl::fmax(rowmax, lmax[g]);
            }
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (lid == 0) scale[row] = s;
            if constexpr (MODE == 3 || MODE == 7) {
                if (lane == 0) scale[row] = s + v[0];  // keep v live
                return;
            }
            if constexpr (MODE == 6)
                store_reg_i32(qout + row * K + off, v, 1.0f / s);
            else
                store_reg(qout + row * K + off, v, 1.0f / s);
        });
    }).wait();
}

// Variant: two rows per work-group (1024 threads), so one barrier and one
// local-memory reduce serve two rows instead of one. The per-thread work and
// the thread->(row, group, register) mapping are unchanged.
static void run_two_rows(queue &q, const int32_t *acc, int8_t *qout, float *scale,
                         const float *x_scale, const float *w_scale, size_t M, size_t K) {
    const size_t n_groups = K / GS;
    const size_t half_wg = n_groups * SUBG;   // 512: one row
    const size_t wg = 2 * half_wg;            // two rows
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    q.submit([&](handler &h) {
        sycl::local_accessor<float, 1> lmax(range<1>(2 * n_groups), h);
        h.parallel_for(nd_range<1>(range<1>(((M + 1) / 2) * wg), range<1>(wg)),
                       [=](nd_item<1> it) {
            const size_t gid = it.get_global_id(0);
            const size_t lid = it.get_local_id(0);
            const size_t which = lid / half_wg;      // 0 or 1
            // odd M: the last work-group's second row does not exist; clamp it
            // onto the last row instead of returning, or the group barrier
            // would be reached by only half the work-group.
            const size_t row = sycl::min((gid / wg) * 2 + which, M - 1);
            const size_t l = lid % half_wg;
            const size_t sgid = l / SUBG;
            const size_t lane = l % SUBG;
            const size_t off = sgid * GS + lane * 16;
            const int32_t *ar = acc + row * (2 * K);

            F16 v = load_dequant_situ_reg(ar + off, ar + K + off, w_scale + off,
                                          w_scale + K + off, x_scale[row]);
            v = inreg(v, c0, c1, c2, c3, d0, d1, d2, d3);
            const auto sg = it.get_sub_group();
            v = bfly_xor(sg, v, 1u, 2u);
            v = bfly_xor(sg, v, 4u, 8u);

            const F16 a = sycl::fabs(v);
            F16 r0 = sycl::max(a, a.swizzle<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>());
            F16 r1 = sycl::max(r0, r0.swizzle<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>());
            F16 r2 = sycl::max(r1, r1.swizzle<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>());
            F16 r3 = sycl::max(r2, r2.swizzle<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>());
            const uint32_t me = static_cast<uint32_t>(sg.get_local_linear_id());
            float vm = r3[0];
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 8u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 4u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 2u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 1u));
            if (lane == 0) lmax[which * n_groups + sgid] = vm;

            sycl::group_barrier(it.get_group());
            float rowmax = 0.f;
            for (size_t g = 0; g < n_groups; ++g)
                rowmax = sycl::fmax(rowmax, lmax[which * n_groups + g]);
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (l == 0) scale[row] = s;
            store_reg(qout + row * K + off, v, 1.0f / s);
        });
    }).wait();
}

// Variant: the row max through sycl::reduce_over_group instead of local memory.
template <int MODE>
static void run_reduce(queue &q, const int32_t *acc, int8_t *qout, float *scale,
                       const float *x_scale, const float *w_scale, size_t M, size_t K) {
    const size_t n_groups = K / GS;
    const size_t wg = n_groups * SUBG;
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    q.submit([&](handler &h) {
        sycl::local_accessor<float, 1> lmax(range<1>(n_groups), h);
        h.parallel_for(nd_range<1>(range<1>(M * wg), range<1>(wg)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) / wg;
            const size_t lid = it.get_local_id(0);
            const size_t sgid = lid / SUBG;
            const size_t lane = lid % SUBG;
            const size_t off = sgid * GS + lane * 16;
            const int32_t *ar = acc + row * (2 * K);

            F16 v = load_dequant_situ_reg(ar + off, ar + K + off, w_scale + off,
                                          w_scale + K + off, x_scale[row]);
            v = inreg(v, c0, c1, c2, c3, d0, d1, d2, d3);
            const auto sg = it.get_sub_group();
            v = bfly_xor(sg, v, 1u, 2u);
            v = bfly_xor(sg, v, 4u, 8u);

            const F16 a = sycl::fabs(v);
            F16 r0 = sycl::max(a, a.swizzle<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>());
            F16 r1 = sycl::max(r0, r0.swizzle<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>());
            F16 r2 = sycl::max(r1, r1.swizzle<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>());
            F16 r3 = sycl::max(r2, r2.swizzle<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>());
            float rowmax;
            if constexpr (MODE == 0) {
                const uint32_t me = static_cast<uint32_t>(sg.get_local_linear_id());
                float vm = r3[0];
                vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 8u));
                vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 4u));
                vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 2u));
                vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 1u));
                if (lane == 0) lmax[sgid] = vm;
                sycl::group_barrier(it.get_group());
                rowmax = 0.f;
                for (size_t g = 0; g < n_groups; ++g) rowmax = sycl::fmax(rowmax, lmax[g]);
            } else {
                const float vm = r3[0];
                rowmax = sycl::reduce_over_group(it.get_group(), vm, sycl::maximum<>());
            }
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (lid == 0) scale[row] = s;
            store_reg(qout + row * K + off, v, 1.0f / s);
        });
    }).wait();
}

// Same kernel, R ConvRot groups per lane instead of 1: the row's work-group
// shrinks by R (512 -> 256 -> 128 threads at K=8192), so the row-wide max is
// over n_groups/R local maxima and the barrier is over R times fewer threads.
// Every element still does the same dequant -> situ -> Hadamard -> scale, and
// the row max is a max over the same set of values, so the output must be
// bit-identical (max is order-independent for these non-negative magnitudes).
template <int R>
static void run_regs(queue &q, const int32_t *acc, int8_t *qout, float *scale,
                     const float *x_scale, const float *w_scale, size_t M, size_t K) {
    const size_t n_groups = K / GS;
    const size_t per_wg = n_groups / R;
    const size_t wg = per_wg * SUBG;
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    q.submit([&](handler &h) {
        sycl::local_accessor<float, 1> lmax(range<1>(per_wg), h);
        h.parallel_for(nd_range<1>(range<1>(M * wg), range<1>(wg)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) / wg;
            const size_t lid = it.get_local_id(0);
            const size_t sgid = lid / SUBG;
            const size_t lane = lid % SUBG;
            const int32_t *ar = acc + row * (2 * K);
            const auto sg = it.get_sub_group();

            F16 v[R];
            float vm = 0.f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const size_t off = (sgid + r * per_wg) * GS + lane * 16;
                const I32 hi = reinterpret_cast<const I32 *>(ar + off)[0];
                const I32 gi = reinterpret_cast<const I32 *>(ar + K + off)[0];
                const F16 hw = reinterpret_cast<const F16 *>(w_scale + off)[0];
                const F16 gw = reinterpret_cast<const F16 *>(w_scale + K + off)[0];
                v[r] = situ_fast(dequant_reg(hi, hw, x_scale[row]),
                                 dequant_reg(gi, gw, x_scale[row]));
                v[r] = inreg(v[r], c0, c1, c2, c3, d0, d1, d2, d3);
                v[r] = bfly_xor(sg, v[r], 1u, 2u);
                v[r] = bfly_xor(sg, v[r], 4u, 8u);
                const F16 a = sycl::fabs(v[r]);
                F16 r0 = sycl::max(a, a.swizzle<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>());
                F16 r1 = sycl::max(r0, r0.swizzle<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>());
                F16 r2 = sycl::max(r1, r1.swizzle<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>());
                F16 r3 = sycl::max(r2, r2.swizzle<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>());
                vm = sycl::fmax(vm, r3[0]);
            }
            const uint32_t me = static_cast<uint32_t>(sg.get_local_linear_id());
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 8u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 4u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 2u));
            vm = sycl::fmax(vm, sycl::select_from_group(sg, vm, me ^ 1u));
            if (lane == 0) lmax[sgid] = vm;

            sycl::group_barrier(it.get_group());
            float rowmax = 0.f;
            for (size_t g = 0; g < per_wg; ++g) rowmax = sycl::fmax(rowmax, lmax[g]);
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (lid == 0) scale[row] = s;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const size_t off = (sgid + r * per_wg) * GS + lane * 16;
                store_reg(qout + row * K + off, v[r], 1.0f / s);
            }
        });
    }).wait();
}

int main(int argc, char **argv) {
    queue q;
    {
        auto d = q.get_device();
        std::printf("device      : %s\n", d.get_info<info::device::name>().c_str());
        std::printf("L2 cache    : %.1f MB\n",
                    d.get_info<info::device::global_mem_cache_size>() / 1048576.0);
        std::printf("local mem   : %.1f KB\n", d.get_info<info::device::local_mem_size>() / 1024.0);
        std::printf("max WG size : %zu\n", (size_t)d.get_info<info::device::max_work_group_size>());
    }
    const size_t M = argc > 1 ? strtoul(argv[1], 0, 10) : 3973;
    const size_t K = argc > 2 ? strtoul(argv[2], 0, 10) : 8192;
    // Production runs on torch's device (VRAM) USM; malloc_shared may land
    // somewhere else. PROBE_MEM=device replicates production.
    const char *memarg = argc > 3 ? argv[3] : getenv("PROBE_MEM");
    const bool dev_mem = memarg && std::strcmp(memarg, "device") == 0;

    int32_t *acc;
    int8_t *out;
    float *sc, *xs, *ws;
    int32_t *h_acc;
    float *h_xs, *h_ws;
    if (dev_mem) {
        acc = malloc_device<int32_t>(M * 2 * K, q);
        out = malloc_device<int8_t>(M * K, q);
        sc = malloc_device<float>(M, q);
        xs = malloc_device<float>(M, q);
        ws = malloc_device<float>(2 * K, q);
        h_acc = new int32_t[M * 2 * K];
        h_xs = new float[M];
        h_ws = new float[2 * K];
    } else {
        acc = malloc_shared<int32_t>(M * 2 * K, q);
        out = malloc_shared<int8_t>(M * K, q);
        sc = malloc_shared<float>(M, q);
        xs = malloc_shared<float>(M, q);
        ws = malloc_shared<float>(2 * K, q);
        h_acc = acc;
        h_xs = xs;
        h_ws = ws;
    }
    // plausible magnitudes: int32 accumulators of an int8 GEMM, scales ~1e-3
    // PROBE_DATA=sat replicates bench_ffn_fusion.py exactly (acc +-1M,
    // ws 1e-3..1.1e-2, xs 1e-2..6e-2 -> dequant +-660, situ saturated).
    const char *datarg = getenv("PROBE_DATA");
    const bool sat = datarg && std::strcmp(datarg, "sat") == 0;
    uint32_t rng = 12345u;
    for (size_t i = 0; i < M * 2 * K; i++) {
        rng = rng * 1103515245u + 12345u;
        if (sat)
            h_acc[i] = int32_t(rng % 2097152u) - 1048576;  // +-2^20
        else
            h_acc[i] = int32_t((rng >> 9) % 4001) - 2000;
    }
    for (size_t i = 0; i < 2 * K; i++) {
        rng = rng * 1103515245u + 12345u;
        if (sat) {
            h_ws[i] = 0.001f + 0.010f * float(rng & 0xFFFF) / 65535.0f;
        } else {
            const uint32_t bits = 0x3A000000u | ((rng >> 16) & 0x007FFFFFu);  // ~1e-4
            std::memcpy(&h_ws[i], &bits, 4);
        }
    }
    for (size_t i = 0; i < M; i++)
        h_xs[i] = sat ? 0.01f + 0.05f * float((i * 2654435761u) & 0xFFFF) / 65535.0f
                      : 0.002f;
    if (dev_mem) {
        q.memcpy(acc, h_acc, sizeof(int32_t) * M * 2 * K);
        q.memcpy(xs, h_xs, sizeof(float) * M);
        q.memcpy(ws, h_ws, sizeof(float) * 2 * K);
        q.memset(out, 0, M * K);
        q.wait();
    } else {
        std::memset(out, 0, M * K);
    }

    const double mb_in = double(M) * 2 * K * 4 / 1048576.0;
    const double mb_out = double(M) * K / 1048576.0;
    std::printf("\nM=%zu K=%zu  in %.1f MB (int32)  out %.1f MB (int8)  mem=%s data=%s\n\n", M, K,
                mb_in, mb_out, dev_mem ? "device" : "shared", sat ? "sat" : "small");

    auto bench = [&](const char *nm, auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < 3; i++) fn();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 12; i++) fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count() / 12);
        }
        std::printf("%-42s %7.4f ms   %6.1f GB/s\n", nm, best,
                    (mb_in + mb_out) / 1024.0 / (best / 1000.0));
        return best;
    };

    double a = bench("0 production", [&] { run<0>(q, acc, out, sc, xs, ws, M, K); });
    double b = bench("1 minus situ", [&] { run<1>(q, acc, out, sc, xs, ws, M, K); });
    double c = bench("2 minus row-max/barrier", [&] { run<2>(q, acc, out, sc, xs, ws, M, K); });
    double d = bench("3 minus int8 store", [&] { run<3>(q, acc, out, sc, xs, ws, M, K); });
    double e = bench("4 minus Hadamard", [&] { run<4>(q, acc, out, sc, xs, ws, M, K); });
    double f = bench("5 loads only", [&] { run<5>(q, acc, out, sc, xs, ws, M, K); });
    double g = bench("6 int32-vector pack", [&] { run<6>(q, acc, out, sc, xs, ws, M, K); });
    double i = bench("7 minus situ, minus store", [&] { run<7>(q, acc, out, sc, xs, ws, M, K); });
    double j = bench("8 minus situ, minus Hadamard", [&] { run<8>(q, acc, out, sc, xs, ws, M, K); });
    double k = bench("10 situ without tanh", [&] { run<10>(q, acc, out, sc, xs, ws, M, K); });
    double l = bench("11 situ without sigmoid", [&] { run<11>(q, acc, out, sc, xs, ws, M, K); });
    double m = bench("12 situ without 1/(1+e)", [&] { run<12>(q, acc, out, sc, xs, ws, M, K); });
    double n = bench("13 situ with expm1 tanh", [&] { run<13>(q, acc, out, sc, xs, ws, M, K); });
    double o = bench("14 two rows per work-group (1024 th)", [&] {
        run_two_rows(q, acc, out, sc, xs, ws, M, K);
    });
    double r2 = bench("16 two groups per lane (256 th)", [&] {
        run_regs<2>(q, acc, out, sc, xs, ws, M, K);
    });
    double r4 = bench("17 four groups per lane (128 th)", [&] {
        run_regs<4>(q, acc, out, sc, xs, ws, M, K);
    });
    double r8 = bench("18 eight groups per lane (64 th)", [&] {
        run_regs<8>(q, acc, out, sc, xs, ws, M, K);
    });
    // Mode 15 replaces the local-memory row max with one
    // sycl::reduce_over_group over the whole work-group. It used to segfault
    // this process (twice, intermittently) and was removed from the sweep.
    // Re-tested 2026-10-21: thousands of launches, no crash, and it is the
    // fastest row-max form here -- so it is back, opt-in, together with a
    // bit-exactness check against mode 0. A regression shows up as a crash or
    // a diff in this probe, never in production.
    //
    //   PROBE_REDUCE=1 ./bin/probe_ffn_situ 3973 8192 device
    double p = 0.0;
    if (std::getenv("PROBE_REDUCE")) {
        std::vector<int8_t> ref(M * K), got(M * K);
        run<0>(q, acc, out, sc, xs, ws, M, K);
        q.memcpy(ref.data(), out, M * K).wait();
        run_reduce<1>(q, acc, out, sc, xs, ws, M, K);
        q.memcpy(got.data(), out, M * K).wait();
        size_t diff = 0;
        for (size_t i = 0; i < M * K; ++i) diff += ref[i] != got[i];
        std::printf("\n  row-max forms: %zu / %zu output bytes differ\n", diff, M * K);

        // Interleaved pairs: the two variants alternate inside the same second,
        // so desktop-clock drift cannot bias the ratio.
        auto timeit = [&](auto fn, int reps) {
            for (int i = 0; i < 3; ++i) fn();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < reps; ++i) fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        };
        std::vector<double> ratios;
        for (int r = 0; r < 15; ++r) {
            const double t0 = timeit([&] { run<0>(q, acc, out, sc, xs, ws, M, K); }, 20);
            const double t15 = timeit([&] { run_reduce<1>(q, acc, out, sc, xs, ws, M, K); }, 20);
            ratios.push_back(t15 / t0);
            std::printf("  pair %2d: mode0 %.4f ms   mode15 %.4f ms\n", r, t0, t15);
        }
        std::sort(ratios.begin(), ratios.end());
        std::printf("  mode15 / mode0 = %.3f (median, range %.3f - %.3f)\n",
                    ratios[ratios.size() / 2], ratios.front(), ratios.back());
        std::fflush(stdout);
        p = bench("15 reduce_over_group row-max",
                  [&] { run_reduce<1>(q, acc, out, sc, xs, ws, M, K); });
    }

    std::printf("\n  situ (tanh/exp chain)      = %+.4f ms\n", a - b);
    std::printf("  row-max + barrier + store  = %+.4f ms  (2 vs 0)\n", c - a);
    std::printf("  int8 store                 = %+.4f ms  (0 vs 3)\n", a - d);
    std::printf("  Hadamard transform         = %+.4f ms  (0 vs 4)\n", a - e);
    std::printf("  int32 pack vs scalar       = %+.4f ms\n", g - a);
    std::printf("  read floor (loads only)    = %.4f ms  (%.0f%% of production)\n", f, 100.0 * f / a);
    std::printf("  read+transform+max         = %.4f ms  (mode 7)\n", i);
    std::printf("  read+dequant+max+store     = %.4f ms  (mode 8)\n", j);
    std::printf("\n  two tanh                   = %+.4f ms  (0 vs 10)\n", a - k);
    std::printf("  exp+reciprocal (sigmoid)   = %+.4f ms  (0 vs 11)\n", a - l);
    std::printf("  reciprocal only            = %+.4f ms  (0 vs 12)\n", a - m);
    std::printf("  expm1 tanh vs sycl::tanh   = %+.4f ms  (0 vs 13)\n", a - n);
    std::printf("  two rows per work-group    = %+.4f ms  (0 vs 14)\n", a - o);
    std::printf("  two groups per lane        = %+.4f ms  (13 vs 16)\n", n - r2);
    std::printf("  four groups per lane       = %+.4f ms  (13 vs 17)\n", n - r4);
    std::printf("  eight groups per lane      = %+.4f ms  (13 vs 18)\n", n - r8);
    if (std::getenv("PROBE_REDUCE"))
        std::printf("  reduce_over_group row max  = %.4f ms (mode 15, opt-in)\n", p);
    return 0;
}
