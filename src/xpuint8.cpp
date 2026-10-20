// xpu-int8 implementation: fused ConvRot rotate+quantize and dequant epilogue.
//
// See include/xpu_kernels/convrot.h for the public API and the algorithm notes.

#include "xpu_kernels/convrot.h"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>

namespace {

using sycl::device;
using sycl::exception;
using sycl::handler;
using sycl::id;
using sycl::nd_item;
using sycl::nd_range;
using sycl::queue;
using sycl::range;
using sycl::sub_group;
using sycl::vec;

using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;
using BF16 = vec<BF, 16>;
using I32 = vec<int32_t, 16>;

std::mutex g_mutex;
std::string g_error;
std::unique_ptr<queue> g_queue;
std::string g_device_name;
int g_convrot_groupsize = 256;
// Limits of the device the queue was created on.  The sub-group-per-group
// kernel needs a work-group that covers a whole row (n_groups * 16 threads)
// and needs the hardware sub-group to be a multiple of 16 so that the
// cross-lane butterflies never leave their 16-lane block.
size_t g_max_work_group = 0;
bool g_subgroup_ok = false;

void set_error(const std::string &e) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_error = e;
}

// ---------------------------------------------------------------------------
// Kernel 1: fused Kronecker-Hadamard rotation + per-row INT8 quantization.
//
// ConvRot's H is kron(H4,...,H4)/sqrt(gs) with
//   H4 = [[ 1, 1, 1,-1],[ 1, 1,-1, 1],[ 1,-1, 1, 1],[-1, 1, 1, 1]]
// A Kronecker product of identical factors applies H4 along each axis of the 4D
// tensor view, and the axis->stride mapping is arbitrary because all factors are
// the same matrix. A gs=256 group is held as 16 vec<float,16> registers (16 zmm)
// with element i = 16*w + q, giving
//   i0 = q & 3, i1 = (q>>2) & 3, i2 = w & 3, i3 = w >> 2
// so each axis is a plain 4-point butterfly:
//   axis 0 : in-register, stride 1 -> 4 broadcast-swizzles + sign vectors
//   axis 1 : in-register, stride 4 -> 4 broadcast-swizzles + sign vectors
//   axis 2 : across registers {4g+0..4g+3}
//   axis 3 : across registers {c, c+4, c+8, c+12}
// Each butterfly is (H4/2); four give (H4/2)^4 = 1/16 = 1/sqrt(256).
// Cost: ~2 vector ops per element instead of 256 MACs per element.
// ---------------------------------------------------------------------------

constexpr int GS = 256;
constexpr int NREG = GS / 16;
constexpr int SUBG = 16;

const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};

// H4 column c expanded over lanes (digit0 = q & 3), pre-scaled by 1/2.
F16 h4_col(int c, bool by_digit1) {
    F16 v;
    for (int t = 0; t < 16; t++)
        v[t] = 0.5f * H4[by_digit1 ? ((t >> 2) & 3) : (t & 3)][c];
    return v;
}

U16 even_mask() {
    U16 m;
    for (int t = 0; t < 16; t++) m[t] = (t & 1) ? 0u : 0xFFFFFFFFu;
    return m;
}

// Load 16 packed bf16 (one 32-byte fetch) into a vec<float,16>.
//
// A 64-byte load path (8 fetches instead of 16, reconstructing each pair of
// registers with a swizzle+mask interleave instead of 16 scalar lane writes)
// was implemented and measured. A paired A/B over 7 interleaved rounds gives
// 0.995x -- indistinguishable -- with bit-identical output, so the simpler
// 32-byte form is kept. See debug/probe_load_ab.py.
F16 load_reg(const uint16_t *p) {
    const vec<uint32_t, 8> u = reinterpret_cast<const vec<uint32_t, 8> *>(p)[0];
    U16 w;
    for (int j = 0; j < 8; j++) {
        w[2 * j] = u[j] << 16;             // low bf16 of the word
        w[2 * j + 1] = u[j] & 0xFFFF0000u; // high bf16 of the word
    }
    return sycl::bit_cast<F16>(w);
}

// MossImage's soft-capped SwiGLU ("situ") on a bf16-rounded (h, g) pair:
//     tanh(g/4) * sigmoid(g) * tanh(h/25)
// with h in the *first* half and g in the second -- the opposite order to
// swiglu, because the model chunks `hidden, gate`.  Each step is rounded to
// bf16 exactly like the eager torch chain, including rounding the division
// results before tanh (torch's `gate / 4.0` is a bf16 tensor).  Without the
// division rounds the activation is still within one ULP, but the row maximum
// can land on a different quantizer step, shifting a whole row's int8 values;
// with them the quantizer is bit-identical to the eager path.
// tanh(z) = expm1(2z) / (expm1(2z) + 2). Algebraically identical to tanh, and
// after the bf16 rounding below it is bit-identical to sycl::tanh for every
// finite bf16 input (checked over all 65536 encodings, debug/probe_tanh_exact).
// The reason to use it: sycl::tanh has a data-dependent slow path. On the
// FFN's saturated activations the two calls cost 3.5x more than on small ones
// (0.71 vs 0.21 ms of an M=3974, K=8192 kernel), while expm1 costs the same
// at every magnitude -- 0.60 ms/call on saturated data, nothing on small data.
//
// |z| is clamped to 20 before expm1: expm1(40) is finite but expm1(2z)
// overflows for |z| > 44 and inf/inf would be NaN. tanh(20) is 1 - 2e-18,
// i.e. exactly 1.0f, so the clamp is exact. The sign is reapplied with a
// bit-or rather than copysign so the vector form cannot be lowered into a
// branch. (Only non-finite input differs: sycl::tanh(inf) is NaN here, the
// clamped form returns 1; situ's arguments come from a finite GEMM.)
F16 tanh_fast(F16 z) {
    const F16 a = sycl::fmin(sycl::fabs(z), F16(20.0f));
    const F16 m = sycl::expm1(2.0f * a);
    const F16 r = m / (m + 2.0f);
    const U16 s = sycl::bit_cast<U16>(r) | (sycl::bit_cast<U16>(z) & U16(0x80000000u));
    return sycl::bit_cast<F16>(s);
}

F16 situ_reg(F16 h, F16 g) {
    F16 tg = tanh_fast((g / 4.0f).template convert<BF>().template convert<float>());
    tg = tg.template convert<BF>().template convert<float>();
    F16 sg = 1.0f / (1.0f + sycl::exp(-g));
    sg = sg.template convert<BF>().template convert<float>();
    F16 th = tanh_fast((h / 25.0f).template convert<BF>().template convert<float>());
    th = th.template convert<BF>().template convert<float>();
    F16 p = tg * sg;
    p = p.template convert<BF>().template convert<float>();
    return (p * th).template convert<BF>().template convert<float>();
}

template <int ACT>
F16 load_activated_reg(const uint16_t *gate, const uint16_t *up) {
    if constexpr (ACT == 0) {
        return load_reg(gate);
    } else if constexpr (ACT == 1) {
        const F16 x = load_reg(gate);
        const F16 x2 = x * x;
        const F16 y = 0.5f * x * (1.0f + sycl::tanh(0.7978845608028654f * x * (1.0f + 0.044715f * x2)));
        return y.template convert<BF>().template convert<float>();
    } else if constexpr (ACT == 2) {
        const F16 g = load_reg(gate);
        const F16 u = load_reg(up);
        const F16 silu = (g / (1.0f + sycl::exp(-g))).template convert<BF>().template convert<float>();
        return (silu * u).template convert<BF>().template convert<float>();
    } else {
        // ACT == 3: MossImage's situ, see situ_reg.
        return situ_reg(load_reg(gate), load_reg(up));
    }
}

// Load one 16-element register of the hidden half and the matching register of
// the gate half straight from the int32 GEMM accumulators, dequantize both, and
// apply MossImage's situ activation.
//
// This is the dequant epilogue and the situ quantizer fused into one pass: the
// up-projection GEMM writes int32 [M, 2K]; instead of dequantizing that to
// bf16 [M, 2K] and reading it back, the quantizer reads the accumulator.
//
// The arithmetic is bit-identical to the unfused pair: launch_dequant_epilogue
// computes `float(acc) * (w_scale * x_scale)` in fp32 and stores the bf16
// rounding of it, so the situ input here is rounded to bf16 the same way
// before the activation, and situ itself keeps exactly the steps of
// load_activated_reg<3>.
F16 load_dequant_situ_reg(const int32_t *hidden, const int32_t *gate,
                          const float *hidden_w, const float *gate_w, float x_scale) {
    const I32 hi = reinterpret_cast<const I32 *>(hidden)[0];
    const I32 gi = reinterpret_cast<const I32 *>(gate)[0];
    const F16 hw = reinterpret_cast<const F16 *>(hidden_w)[0];
    const F16 gw = reinterpret_cast<const F16 *>(gate_w)[0];
    // dequant, then the bf16 round that the bf16 store/reload used to impose
    F16 h = (F16(hi.convert<float>()) * (hw * x_scale))
                .template convert<BF>().template convert<float>();
    F16 g = (F16(gi.convert<float>()) * (gw * x_scale))
                .template convert<BF>().template convert<float>();
    return situ_reg(h, g);
}

// Round-to-nearest and clamp into the int8 range, entirely in vector registers.
F16 quantize_vec(F16 v, float inv) {
    return sycl::min(sycl::max(sycl::rint(v * inv), F16(-127.f)), F16(127.f));
}

void store_reg(int8_t *p, F16 v, float inv) {
    const F16 r = quantize_vec(v, inv);
    uint32_t b[4] = {0, 0, 0, 0};
    for (int t = 0; t < 16; t++)
        b[t >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(
                         static_cast<int8_t>(static_cast<int32_t>(r[t]))))
                     << (8 * (t & 3));
    std::memcpy(p, b, 16);
}

// Elementwise 4-point butterfly across four registers: V = (H4/2) * U.
void bfly4(F16 &a0, F16 &a1, F16 &a2, F16 &a3) {
    const F16 p = a0 + a1, m = a0 - a1, q = a2 + a3, n = a2 - a3;
    a0 = (p + n) * 0.5f;
    a1 = (p - n) * 0.5f;
    a2 = (m + q) * 0.5f;
    a3 = (q - m) * 0.5f;
}

// Axes 0 and 1, which act inside a single register. Written as an explicit
// accumulation chain so only one product temporary is live at a time; the
// one-expression form needs four and spills.
F16 inreg(F16 v, const F16 &c0, const F16 &c1, const F16 &c2, const F16 &c3,
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

// One work-item handles at most NGRP ConvRot groups: W holds them all, row
// abs-max accumulates across them, then the sub-group reduce gives the row
// scale and all groups are quantized.
//
// NGRP is a template parameter so the W array has compile-time indices. A
// runtime-indexed W[g][w] is demoted to scratch memory, which costs ~35% on
// this kernel; the one-group case (K <= 4096, i.e. every shape that matters)
// is written as a plain F16[NREG] for that reason.
template <int NGRP, int ACT>
void convrot_body(nd_item<1> it, const uint16_t *x, int8_t *qout, float *scale, size_t M,
                  size_t K, size_t n_groups, const F16 &c0, const F16 &c1, const F16 &c2,
                  const F16 &c3, const F16 &d0, const F16 &d1, const F16 &d2,
                  const F16 &d3) {
    const size_t row = it.get_global_id(0) >> 4;
    const size_t lane = it.get_local_id(0) & 15;
    const size_t input_K = ACT >= 2 ? K * 2 : K;  // swiglu (2) and situ (3) read a paired input
    const uint16_t *xr = x + row * input_K;
    const uint16_t *ur = ACT >= 2 ? xr + K : nullptr;

    F16 amaxv(0.f);

    // One group's load + 4-axis Kronecker transform + abs-max contribution.
    auto rotate_group = [&](F16 *W, const uint16_t *p, const uint16_t *up) {
        // Load and transform one axis-2 group at a time so that load latency
        // overlaps the butterfly work instead of being exposed once per group.
        for (int b = 0; b < NREG; b += 4) {
            for (int j = 0; j < 4; j++)
                W[b + j] = load_activated_reg<ACT>(p + (b + j) * 16,
                                                    up ? up + (b + j) * 16 : nullptr);
            for (int j = 0; j < 4; j++)
                W[b + j] = inreg(W[b + j], c0, c1, c2, c3, d0, d1, d2, d3);
            bfly4(W[b], W[b + 1], W[b + 2], W[b + 3]);
        }
        for (int c = 0; c < 4; c++) bfly4(W[c], W[c + 4], W[c + 8], W[c + 12]);
        for (int w = 0; w < NREG; ++w) amaxv = sycl::max(amaxv, sycl::fabs(W[w]));
    };

    // NGRP == 1 (K <= 4096) uses a flat F16[NREG]; the multi-group path keeps
    // the 2-D form because its g index is unrolled anyway.
    F16 W1[NREG];
    F16 WN[NGRP == 1 ? 1 : NGRP][NREG];

    if (NGRP == 1) {
        if (lane < n_groups)
            rotate_group(W1, xr + lane * GS, ur ? ur + lane * GS : nullptr);
    } else {
        for (int g = 0; g < NGRP; ++g) {
            const size_t grp = static_cast<size_t>(g) * SUBG + lane;
            if (grp < n_groups)
                rotate_group(WN[g], xr + grp * GS, ur ? ur + grp * GS : nullptr);
        }
    }

    // horizontal max, then sub-group max (one row = SUBG lanes)
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
            for (int w = 0; w < NREG; ++w) store_reg(p + w * 16, W1[w], inv);
        }
    } else {
        for (int g = 0; g < NGRP; ++g) {
            const size_t grp = static_cast<size_t>(g) * SUBG + lane;
            if (grp >= n_groups) continue;
            int8_t *p = qout + row * K + grp * GS;
            for (int w = 0; w < NREG; ++w) store_reg(p + w * 16, WN[g][w], inv);
        }
    }
}

template <int G, int ACT>
void launch_convrot_quant(queue &q, const uint16_t *x, int8_t *qout, float *scale,
                          size_t M, size_t K, bool wait) {
    const size_t n_groups = K / GS;
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    auto event = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M * SUBG), range<1>(SUBG)),
                       [=](nd_item<1> it) {
                           convrot_body<G, ACT>(it, x, qout, scale, M, K, n_groups, c0, c1,
                                                c2, c3, d0, d1, d2, d3);
                       });
    });
    if (wait) event.wait();
}

// ---------------------------------------------------------------------------
// Kernel 1b: the same transform, with a ConvRot group owned by a whole
// sub-group instead of by a single lane.
//
// The kernel above maps one work-item to one 256-element ConvRot group, so
// all 16 registers of a group live in one lane.  That lane reads 512 B
// contiguous and writes 256 B contiguous, but the 16 lanes of the sub-group
// are 512 B apart: at a single load instruction the warp touches 16
// addresses 512 B apart and each 32 B request sits alone in a 128 B line.
// Measured at M=3952 K=3072 that layout runs at 67 GB/s.
//
// Here lane L of a sub-group holds register L of the group.  The warp's
// single load then covers 512 B fully contiguous and its single store covers
// 256 B fully contiguous, and because a lane now holds exactly ONE register
// the transformed value survives the row-scale reduction -- the kernel still
// reads x once, writes q once, and transforms once.
//
// The two cross-register Kronecker axes become cross-lane butterflies: axis 2
// (registers 4g..4g+3) is xor masks 1 and 2, axis 3 (registers c, c+4, c+8,
// c+12) is xor masks 4 and 8.  A 4-point Hadamard butterfly over the lanes
// {L, L^m1, L^m2, L^m1^m2} decomposes as
//     A = v + partner(v, m1)
//     B = v - partner(v, m1)
//     out = (A + partner(B, m2)) * 0.5
// which reproduces bfly4() exactly.
//
// A row spans n_groups groups, so the work-group covers a whole row
// (n_groups sub-groups of 16 lanes) and the per-group maxima go through
// local memory, one barrier, and every lane reduces them itself.
//
// Measured at M=3952: K=3072 0.541 -> 0.130 ms (4.2x), K=8192 3.83 -> 0.345
// ms (11.1x).  Bit-identical output to the kernel above at both shapes.
// ---------------------------------------------------------------------------

// 4-point Hadamard butterfly across four lanes of a sub-group.
inline F16 bfly_xor(sub_group sg, F16 v, uint32_t m1, uint32_t m2) {
    const uint32_t me = static_cast<uint32_t>(sg.get_local_linear_id());
    const F16 p = sycl::select_from_group(sg, v, me ^ m1);
    const F16 A = v + p;
    const F16 B = v - p;
    const F16 Bp = sycl::select_from_group(sg, B, me ^ m2);
    return (A + Bp) * 0.5f;
}

template <int ACT>
void launch_convrot_quant_warp(queue &q, const uint16_t *x, int8_t *qout, float *scale,
                                size_t M, size_t K, bool wait) {
    const size_t n_groups = K / GS;
    const size_t wg = n_groups * SUBG;  // one work-group per row
    const size_t input_K = ACT >= 2 ? K * 2 : K;  // swiglu (2) and situ (3) read a paired input
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    auto event = q.submit([&](handler &h) {
        sycl::local_accessor<float, 1> lmax(range<1>(n_groups), h);
        h.parallel_for(nd_range<1>(range<1>(M * wg), range<1>(wg)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) / wg;
            const size_t lid = it.get_local_id(0);
            const size_t sgid = lid / SUBG;  // which group of the row
            const size_t lane = lid % SUBG;  // which register of the group
            const uint16_t *xr = x + row * input_K;
            const uint16_t *ur = ACT >= 2 ? xr + K : nullptr;
            const uint16_t *p = xr + sgid * GS + lane * 16;
            const uint16_t *up = ur ? ur + sgid * GS + lane * 16 : nullptr;

            F16 v = load_activated_reg<ACT>(p, up);
            v = inreg(v, c0, c1, c2, c3, d0, d1, d2, d3);
            const auto sg = it.get_sub_group();
            v = bfly_xor(sg, v, 1u, 2u);  // axis 2
            v = bfly_xor(sg, v, 4u, 8u);  // axis 3

            // abs-max of this lane's register, then across the sub-group
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
            if (lane == 0) lmax[sgid] = vm;

            sycl::group_barrier(it.get_group());
            float rowmax = 0.f;
            for (size_t g = 0; g < n_groups; ++g) rowmax = sycl::fmax(rowmax, lmax[g]);
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (lid == 0) scale[row] = s;
            store_reg(qout + row * K + sgid * GS + lane * 16, v, 1.0f / s);
        });
    });
    if (wait) event.wait();
}

// ---------------------------------------------------------------------------
// Kernel 1c: dequantize the up-projection accumulator, apply situ, rotate and
// quantize -- one pass instead of dequant -> bf16 -> rotate+quantize.
//
// Same work decomposition as launch_convrot_quant_warp: a row is owned by a
// whole work-group, and lane L of a sub-group holds register L (16 values) of
// its ConvRot group. acc is int32 [M, 2K]; the hidden half is the first K
// columns and the gate half the second K (the model chunks `hidden, gate`).
// x_scale is the per-row scale of the up-projection's input, w_scale the
// per-column scale of its output.
// ---------------------------------------------------------------------------
void launch_dequant_situ_quant_warp(queue &q, const int32_t *acc, int8_t *qout,
                                     float *scale, const float *x_scale,
                                     const float *w_scale, size_t M, size_t K,
                                     bool wait) {
    const size_t n_groups = K / GS;
    const size_t wg = n_groups * SUBG;  // one work-group per row
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    auto event = q.submit([&](handler &h) {
        sycl::local_accessor<float, 1> lmax(range<1>(n_groups), h);
        h.parallel_for(nd_range<1>(range<1>(M * wg), range<1>(wg)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) / wg;
            const size_t lid = it.get_local_id(0);
            const size_t sgid = lid / SUBG;  // which group of the row
            const size_t lane = lid % SUBG;  // which register of the group
            const size_t off = sgid * GS + lane * 16;
            const int32_t *ar = acc + row * (2 * K);

            F16 v = load_dequant_situ_reg(ar + off, ar + K + off,
                                          w_scale + off, w_scale + K + off,
                                          x_scale[row]);
            v = inreg(v, c0, c1, c2, c3, d0, d1, d2, d3);
            const auto sg = it.get_sub_group();
            v = bfly_xor(sg, v, 1u, 2u);  // axis 2
            v = bfly_xor(sg, v, 4u, 8u);  // axis 3

            // abs-max of this lane's register, then across the sub-group
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
            if (lane == 0) lmax[sgid] = vm;

            sycl::group_barrier(it.get_group());
            float rowmax = 0.f;
            for (size_t g = 0; g < n_groups; ++g) rowmax = sycl::fmax(rowmax, lmax[g]);
            const float s = sycl::fmax(rowmax / 127.0f, 1e-30f);
            if (lid == 0) scale[row] = s;
            store_reg(qout + row * K + off, v, 1.0f / s);
        });
    });
    if (wait) event.wait();
}

// ---------------------------------------------------------------------------
// Kernel 2: rotation only, bf16 -> bf16.  Used by the BF16 path, where the
// weights are already stored dequantized-and-pre-rotated, so the activation
// needs the same rotation but no quantization.
//
// This is the same transform as kernel 1 with the row abs-max, the sub-group
// reduce and the int8 quantize removed. Dropping them matters: without the
// row-scale dependency the axis-3 butterfly can store each 4-register group
// as soon as it is finished, so registers are released progressively and the
// loads of the next group overlap the stores of the current one. That is
// exactly the 0.128 ms-vs-0.402 ms interleaving the rotate+quantize kernel
// cannot reach.
// ---------------------------------------------------------------------------
template <int G>
void launch_rotate_bf16(queue &q, const uint16_t *x, BF *xr, size_t M, size_t K,
                        bool wait) {
    const size_t n_groups = K / GS;
    const F16 c0 = h4_col(0, false), c1 = h4_col(1, false), c2 = h4_col(2, false),
              c3 = h4_col(3, false);
    const F16 d0 = h4_col(0, true), d1 = h4_col(1, true), d2 = h4_col(2, true),
              d3 = h4_col(3, true);

    auto event = q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M * SUBG), range<1>(SUBG)),
                       [=](nd_item<1> it) {
                           const size_t row = it.get_global_id(0) >> 4;
                           const size_t lane = it.get_local_id(0) & 15;
                           const uint16_t *src = x + row * K;
                           BF *dst = xr + row * K;

                           for (int g = 0; g < G; ++g) {
                               const size_t grp = static_cast<size_t>(g) * SUBG + lane;
                               if (grp >= n_groups) continue;
                               const uint16_t *p = src + grp * GS;

                               F16 W[NREG];
                               for (int b = 0; b < NREG; b += 4) {
                                   for (int j = 0; j < 4; j++) W[b + j] = load_reg(p + (b + j) * 16);
                                   for (int j = 0; j < 4; j++)
                                       W[b + j] = inreg(W[b + j], c0, c1, c2, c3, d0, d1, d2, d3);
                                   bfly4(W[b], W[b + 1], W[b + 2], W[b + 3]);
                               }
                               // Axis 3, then one flat convert+store pass. Do NOT interleave the
                               // butterfly with the stores: four butterflies first
                               // and then a straight 16-register store loop is
                               // 1.44x faster (0.315 vs 0.454 ms at M=4096),
                               // because the flat loop keeps the register file
                               // stable while the stores drain.
                               for (int c = 0; c < 4; c++)
                                   bfly4(W[c], W[c + 4], W[c + 8], W[c + 12]);
                               BF *d = dst + grp * GS;
                               for (int w = 0; w < NREG; w++) {
                                   BF16 r;
                                   for (int t = 0; t < 16; t++) r[t] = BF(W[w][t]);
                                   reinterpret_cast<BF16 *>(d + w * 16)[0] = r;
                               }
                           }
                       });
    });
    if (wait) event.wait();
}

// ---------------------------------------------------------------------------
// Kernel 3: fused dequant epilogue.  out = acc * x_scale[m] * w_scale[n] + bias[n]
// ---------------------------------------------------------------------------
void launch_dequant_epilogue(queue &q, const int32_t *acc, BF *out, const float *x_scale,
                             const float *w_scale, const float *bias, size_t M, size_t N,
                             size_t acc_row_stride, bool wait) {
    const size_t nvec = N / 16;
    const int has_bias = bias != nullptr;
    auto event = q.submit([&](handler &h) {
        h.parallel_for(range<2>(M, nvec), [=](id<2> idx) {
            const size_t m = idx[0], nv = idx[1];
            const I32 a = reinterpret_cast<const I32 *>(acc + m * acc_row_stride)[nv];
            const F16 w = reinterpret_cast<const F16 *>(w_scale)[nv];
            F16 v = F16(a.convert<float>()) * (w * x_scale[m]);
            if (has_bias) v = v + reinterpret_cast<const F16 *>(bias)[nv];
            BF16 r;
            for (int t = 0; t < 16; t++) r[t] = BF(v[t]);
            reinterpret_cast<BF16 *>(out + m * N)[nv] = r;
        });
    });
    if (wait) event.wait();
}

bool ensure_queue() {
    if (g_queue) return true;
    try {
        g_queue = std::make_unique<queue>(sycl::gpu_selector_v);
        g_device_name = g_queue->get_device().get_info<sycl::info::device::name>();
        g_max_work_group = g_queue->get_device().get_info<sycl::info::device::max_work_group_size>();
        g_subgroup_ok = true;
        for (auto sg : g_queue->get_device().get_info<sycl::info::device::sub_group_sizes>()) {
            if (sg < SUBG || (sg % SUBG) != 0) g_subgroup_ok = false;
        }
    } catch (const exception &e) {
        set_error(std::string("no SYCL GPU device: ") + e.what());
        g_queue.reset();
        return false;
    }
    return true;
}

template <int ACT>
int convrot_quantize(queue &q, const void *x, int8_t *qout, float *scale, int64_t M,
                     int64_t K, bool wait) {
    if (M <= 0 || K <= 0) {
        set_error("M and K must be positive");
        return 2;
    }
    if (K % GS != 0) {
        set_error("K must be divisible by the ConvRot group size (256)");
        return 3;
    }
    const size_t n_groups = static_cast<size_t>(K) / GS;
    // Preferred path: a ConvRot group per sub-group.  It needs the whole row
    // in one work-group, so it only applies while that fits.
    if (g_subgroup_ok && n_groups * SUBG <= g_max_work_group) {
        try {
            launch_convrot_quant_warp<ACT>(q, static_cast<const uint16_t *>(x), qout, scale,
                                           M, K, wait);
            return 0;
        } catch (const exception &e) {
            set_error(std::string("convrot_quantize (warp): ") + e.what());
            return 5;
        }
    }
    const size_t gpl = (n_groups + SUBG - 1) / SUBG;
    try {
        switch (gpl) {
        case 1: launch_convrot_quant<1, ACT>(q, static_cast<const uint16_t *>(x), qout,
                                             scale, M, K, wait); break;
        case 2: launch_convrot_quant<2, ACT>(q, static_cast<const uint16_t *>(x), qout,
                                             scale, M, K, wait); break;
        case 3: launch_convrot_quant<3, ACT>(q, static_cast<const uint16_t *>(x), qout,
                                             scale, M, K, wait); break;
        case 4: launch_convrot_quant<4, ACT>(q, static_cast<const uint16_t *>(x), qout,
                                             scale, M, K, wait); break;
        case 6: launch_convrot_quant<6, ACT>(q, static_cast<const uint16_t *>(x), qout,
                                             scale, M, K, wait); break;
        case 12: launch_convrot_quant<12, ACT>(q, static_cast<const uint16_t *>(x), qout,
                                               scale, M, K, wait); break;
        default:
            set_error("unsupported group count per sub-group");
            return 4;
        }
    } catch (const exception &e) {
        set_error(std::string("convrot_quantize: ") + e.what());
        return 5;
    }
    return 0;
}

}  // namespace

extern "C" {

int xpuint8_init(void) { return ensure_queue() ? 0 : 1; }

const char *xpuint8_device_name(void) {
    if (!ensure_queue()) return nullptr;
    return g_device_name.c_str();
}

int xpuint8_convrot_groupsize(void) { return g_convrot_groupsize; }

int xpuint8_convrot_quantize(const void *x, int8_t *q, float *scale, int64_t M,
                             int64_t K) {
    if (!ensure_queue()) return 1;
    return convrot_quantize<0>(*g_queue, x, q, scale, M, K, true);
}

int xpuint8_convrot_quantize_stream(const void *sycl_queue, const void *x, int8_t *q,
                                    float *scale, int64_t M, int64_t K) {
    if (!sycl_queue) {
        set_error("SYCL queue pointer must not be null");
        return 6;
    }
    return convrot_quantize<0>(*static_cast<queue *>(const_cast<void *>(sycl_queue)),
                               x, q, scale, M, K, false);
}

int xpuint8_convrot_quantize_act(const void *x, int8_t *q, float *scale, int64_t M,
                                 int64_t K, int32_t activation) {
    if (!ensure_queue()) return 1;
    if (activation == 1) return convrot_quantize<1>(*g_queue, x, q, scale, M, K, true);
    if (activation == 2) return convrot_quantize<2>(*g_queue, x, q, scale, M, K, true);
    if (activation == 3) return convrot_quantize<3>(*g_queue, x, q, scale, M, K, true);
    set_error("activation must be 1 (GELU-tanh), 2 (SwiGLU) or 3 (situ)");
    return 6;
}

int xpuint8_convrot_quantize_act_stream(const void *sycl_queue, const void *x, int8_t *q,
                                        float *scale, int64_t M, int64_t K,
                                        int32_t activation) {
    if (!sycl_queue) {
        set_error("SYCL queue pointer must not be null");
        return 6;
    }
    auto &queue = *static_cast<sycl::queue *>(const_cast<void *>(sycl_queue));
    if (activation == 1) return convrot_quantize<1>(queue, x, q, scale, M, K, false);
    if (activation == 2) return convrot_quantize<2>(queue, x, q, scale, M, K, false);
    if (activation == 3) return convrot_quantize<3>(queue, x, q, scale, M, K, false);
    set_error("activation must be 1 (GELU-tanh), 2 (SwiGLU) or 3 (situ)");
    return 6;
}

// Fused dequantize + situ + rotate + quantize. acc is int32 [M, 2K].
int dequant_act_quantize(queue &q, const void *acc, int8_t *qout, float *scale,
                         const float *x_scale, const float *w_scale, int64_t M,
                         int64_t K, int32_t activation, bool wait) {
    if (M <= 0 || K <= 0) {
        set_error("M and K must be positive");
        return 2;
    }
    if (K % GS != 0) {
        set_error("K must be divisible by the ConvRot group size (256)");
        return 3;
    }
    if (activation != 3) {
        set_error("dequant_act_quantize only supports activation 3 (situ)");
        return 6;
    }
    try {
        launch_dequant_situ_quant_warp(q, static_cast<const int32_t *>(acc), qout,
                                       scale, x_scale, w_scale, M, K, wait);
    } catch (const exception &e) {
        set_error(std::string("dequant_act_quantize: ") + e.what());
        return 5;
    }
    return 0;
}

int xpuint8_dequant_act_quantize(const void *acc, int8_t *q, float *scale,
                                 const float *x_scale, const float *w_scale,
                                 int64_t M, int64_t K, int32_t activation) {
    if (!ensure_queue()) return 1;
    return dequant_act_quantize(*g_queue, acc, q, scale, x_scale, w_scale, M, K,
                                activation, true);
}

int xpuint8_dequant_act_quantize_stream(const void *sycl_queue, const void *acc,
                                        int8_t *q, float *scale,
                                        const float *x_scale, const float *w_scale,
                                        int64_t M, int64_t K, int32_t activation) {
    if (!sycl_queue) {
        set_error("SYCL queue pointer must not be null");
        return 6;
    }
    return dequant_act_quantize(*static_cast<queue *>(const_cast<void *>(sycl_queue)),
                                acc, q, scale, x_scale, w_scale, M, K, activation,
                                false);
}

int convrot_rotate_bf16(queue &q, const void *x, void *xr, int64_t M, int64_t K,
                        bool wait) {
    if (M <= 0 || K <= 0) {
        set_error("M and K must be positive");
        return 2;
    }
    if (K % GS != 0) {
        set_error("K must be divisible by the ConvRot group size (256)");
        return 3;
    }
    const size_t gpl = (static_cast<size_t>(K) / GS + SUBG - 1) / SUBG;
    try {
        switch (gpl) {
        case 1: launch_rotate_bf16<1>(q, static_cast<const uint16_t *>(x),
                                      static_cast<BF *>(xr), M, K, wait); break;
        case 2: launch_rotate_bf16<2>(q, static_cast<const uint16_t *>(x),
                                      static_cast<BF *>(xr), M, K, wait); break;
        case 3: launch_rotate_bf16<3>(q, static_cast<const uint16_t *>(x),
                                      static_cast<BF *>(xr), M, K, wait); break;
        case 4: launch_rotate_bf16<4>(q, static_cast<const uint16_t *>(x),
                                      static_cast<BF *>(xr), M, K, wait); break;
        case 6: launch_rotate_bf16<6>(q, static_cast<const uint16_t *>(x),
                                      static_cast<BF *>(xr), M, K, wait); break;
        case 12: launch_rotate_bf16<12>(q, static_cast<const uint16_t *>(x),
                                        static_cast<BF *>(xr), M, K, wait); break;
        default:
            set_error("unsupported group count per sub-group");
            return 4;
        }
    } catch (const exception &e) {
        set_error(std::string("convrot_rotate_bf16: ") + e.what());
        return 5;
    }
    return 0;
}

int xpuint8_convrot_rotate_bf16(const void *x, void *xr, int64_t M, int64_t K) {
    if (!ensure_queue()) return 1;
    return convrot_rotate_bf16(*g_queue, x, xr, M, K, true);
}

int xpuint8_convrot_rotate_bf16_stream(const void *sycl_queue, const void *x, void *xr,
                                       int64_t M, int64_t K) {
    if (!sycl_queue) {
        set_error("SYCL queue pointer must not be null");
        return 1;
    }
    auto &queue = *static_cast<sycl::queue *>(const_cast<void *>(sycl_queue));
    return convrot_rotate_bf16(queue, x, xr, M, K, false);
}

int xpuint8_dequant_epilogue(const int32_t *acc, void *out, const float *x_scale,
                             const float *w_scale, const float *bias, int64_t M,
                             int64_t N) {
    if (!ensure_queue()) return 1;
    if (M <= 0 || N <= 0 || N % 16 != 0) {
        set_error("M, N must be positive and N must be a multiple of 16");
        return 2;
    }
    try {
        launch_dequant_epilogue(*g_queue, acc, static_cast<BF *>(out), x_scale, w_scale,
                                bias, M, N, N, true);
    } catch (const exception &e) {
        set_error(std::string("dequant_epilogue: ") + e.what());
        return 5;
    }
    return 0;
}

int xpuint8_dequant_epilogue_stream(const void *sycl_queue, const int32_t *acc, void *out,
                                    const float *x_scale, const float *w_scale,
                                    const float *bias, int64_t M, int64_t N) {
    if (!sycl_queue) {
        set_error("SYCL queue pointer must not be null");
        return 1;
    }
    if (M <= 0 || N <= 0 || N % 16 != 0) {
        set_error("M, N must be positive and N must be a multiple of 16");
        return 2;
    }
    try {
        auto &queue = *static_cast<sycl::queue *>(const_cast<void *>(sycl_queue));
        launch_dequant_epilogue(queue, acc, static_cast<BF *>(out), x_scale, w_scale,
                                bias, M, N, N, false);
    } catch (const exception &e) {
        set_error(std::string("dequant_epilogue: ") + e.what());
        return 5;
    }
    return 0;
}

// Row-strided variant: the accumulator is a [M, N] column slice of a wider
// [M, acc_row_stride] buffer (the fused qkv projection), so the dequant can run
// on one third of it without copying.  Same arithmetic, same rounding.
int xpuint8_dequant_epilogue_slice_stream(const void *sycl_queue, const int32_t *acc,
                                          void *out, const float *x_scale,
                                          const float *w_scale, const float *bias,
                                          int64_t M, int64_t N, int64_t acc_row_stride) {
    if (!sycl_queue) {
        set_error("SYCL queue pointer must not be null");
        return 1;
    }
    if (M <= 0 || N <= 0 || N % 16 != 0 || acc_row_stride < N) {
        set_error("M, N must be positive, N a multiple of 16, acc_row_stride >= N");
        return 2;
    }
    try {
        auto &queue = *static_cast<sycl::queue *>(const_cast<void *>(sycl_queue));
        launch_dequant_epilogue(queue, acc, static_cast<BF *>(out), x_scale, w_scale,
                                bias, M, N, static_cast<size_t>(acc_row_stride), false);
    } catch (const exception &e) {
        set_error(std::string("dequant_epilogue_slice: ") + e.what());
        return 5;
    }
    return 0;
}

int xpuint8_dequant_epilogue_slice(const int32_t *acc, void *out, const float *x_scale,
                                   const float *w_scale, const float *bias, int64_t M,
                                   int64_t N, int64_t acc_row_stride) {
    if (!ensure_queue()) return 1;
    if (M <= 0 || N <= 0 || N % 16 != 0 || acc_row_stride < N) {
        set_error("M, N must be positive, N a multiple of 16, acc_row_stride >= N");
        return 2;
    }
    try {
        launch_dequant_epilogue(*g_queue, acc, static_cast<BF *>(out), x_scale, w_scale,
                                bias, M, N, static_cast<size_t>(acc_row_stride), true);
    } catch (const exception &e) {
        set_error(std::string("dequant_epilogue_slice: ") + e.what());
        return 5;
    }
    return 0;
}

const char *xpuint8_last_error(void) { return g_error.c_str(); }

}  // extern "C"
