// Fused ConvRot: Kronecker-Hadamard rotation + per-row INT8 quantization.
// Intel Arc (Xe2) via SYCL.
//
// comfy-kitchen's `eager` backend runs this as 6+ separate global passes
//   reshape -> bmm(M,ng,256,256) -> reshape -> abs -> amax -> div -> round -> clamp -> cast
// costing 1.15 ms at M=4096,K=3072 and dominating the whole INT8 linear layer.
// This kernel does it in one pass at memory-bandwidth speed.
//
// Algorithm
// ---------
// ConvRot's H is kron(H4,...,H4)/sqrt(gs) with
//   H4 = [[ 1, 1, 1,-1],[ 1, 1,-1, 1],[ 1,-1, 1, 1],[-1, 1, 1, 1]]
// A Kronecker product of identical factors applies H4 along each axis of the 4D
// tensor view, and the axis->stride mapping is arbitrary (all factors are equal).
// A gs=256 group is held as 16 vec<float,16> registers (1 KB, 16 zmm) with
// element i = 16*w + q, giving
//   i0 = q & 3, i1 = (q>>2) & 3, i2 = w & 3, i3 = w >> 2
// so each axis is a plain 4-point butterfly:
//   axis 0 : in-register, stride 1  -> 4 broadcast-swizzles + sign vectors
//   axis 1 : in-register, stride 4  -> 4 broadcast-swizzles + sign vectors
//   axis 2 : across registers {4g+0..4g+3}          (4 groups, full-width)
//   axis 3 : across registers {c, c+4, c+8, c+12}   (4 groups, full-width)
// Each butterfly is (H4/2); four give (H4/2)^4 = 1/16 = 1/sqrt(256).
// Cost: ~2 vector ops per element instead of 256 MACs per element for the bmm.
//
// Grid is one sub-group (16 lanes) per row; lane l owns ConvRot groups
// l, l+16, l+32, ... The only cross-lane communication is the row abs-max.

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace xpuint8 {

using sycl::gpu_selector_v;
using sycl::handler;
using sycl::nd_item;
using sycl::nd_range;
using sycl::queue;
using sycl::range;
using sycl::vec;

constexpr int GS = 256;         // ConvRot group size (4 Kronecker factors)
constexpr int NREG = GS / 16;   // 16 registers of 16 floats per group
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;

// lane mask for even/odd lanes of a 16-lane vector
static inline U16 even_mask() {
    U16 m;
    for (int t = 0; t < 16; t++) m[t] = (t & 1) ? 0u : 0xFFFFFFFFu;
    return m;
}
static inline U16 odd_mask() {
    U16 m;
    for (int t = 0; t < 16; t++) m[t] = (t & 1) ? 0xFFFFFFFFu : 0u;
    return m;
}
// H4 column c, expanded over lanes (digit0 = q & 3), pre-scaled by 1/2
static inline F16 h4_col(int c) {
    static const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    F16 v;
    for (int t = 0; t < 16; t++) v[t] = 0.5f * H4[t & 3][c];
    return v;
}
// H4 column c, expanded over lanes (digit1 = (q>>2) & 3), pre-scaled by 1/2
static inline F16 h4_col_d1(int c) {
    static const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    F16 v;
    for (int t = 0; t < 16; t++) v[t] = 0.5f * H4[(t >> 2) & 3][c];
    return v;
}

// Load 16 packed bf16 (one 32-byte fetch) into a vec<float,16>.
static inline F16 load_reg(const uint16_t *p) {
    const vec<uint32_t, 8> u = reinterpret_cast<const vec<uint32_t, 8> *>(p)[0];
    U16 w;
    for (int j = 0; j < 8; j++) {
        w[2 * j] = u[j] << 16;             // low bf16 of the word
        w[2 * j + 1] = u[j] & 0xFFFF0000u; // high bf16 of the word
    }
    return sycl::bit_cast<F16>(w);
}

// Round to nearest and clamp to the int8 range, entirely in vector registers.
static inline F16 quantize_vec(F16 v, float inv) {
    const F16 lim(127.f), nlim(-127.f);
    return sycl::min(sycl::max(sycl::rint(v * inv), nlim), lim);
}

// Store a vec<float,16> as 16 packed int8 (one 16-byte store).
static inline void store_reg(int8_t *p, const F16 &v, float inv) {
    const F16 r = quantize_vec(v, inv);
    uint32_t b[4] = {0, 0, 0, 0};
    for (int t = 0; t < 16; t++)
        b[t >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(
                         static_cast<int8_t>(static_cast<int32_t>(r[t]))))
                     << (8 * (t & 3));
    std::memcpy(p, b, 16);
}

// Elementwise 4-point butterfly across four registers: V = (H4/2) * U.
static inline void bfly4(F16 &a0, F16 &a1, F16 &a2, F16 &a3) {
    const F16 p = a0 + a1, m = a0 - a1, q = a2 + a3, n = a2 - a3;
    a0 = (p + n) * 0.5f;
    a1 = (p - n) * 0.5f;
    a2 = (m + q) * 0.5f;
    a3 = (q - m) * 0.5f;
}

// Rotate one 256-element group in place (16 vec<float,16> registers).
static inline void rotate_group(F16 (&W)[NREG], const F16 &c0, const F16 &c1,
                                const F16 &c2, const F16 &c3, const F16 &d0,
                                const F16 &d1, const F16 &d2, const F16 &d3) {
    // axis 0: digit = q & 3 (stride 1 within each 4-block)
    for (int w = 0; w < NREG; ++w)
        W[w] = c0 * W[w].swizzle<0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12>() +
               c1 * W[w].swizzle<1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13>() +
               c2 * W[w].swizzle<2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14>() +
               c3 * W[w].swizzle<3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15>();

    // axis 1: digit = (q>>2) & 3 (stride 4 within the register)
    for (int w = 0; w < NREG; ++w)
        W[w] = d0 * W[w].swizzle<0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>() +
               d1 * W[w].swizzle<4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7>() +
               d2 * W[w].swizzle<8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11>() +
               d3 * W[w].swizzle<12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15>();

    // axis 2: digit = w & 3
    for (int g = 0; g < 4; g++) bfly4(W[4 * g], W[4 * g + 1], W[4 * g + 2], W[4 * g + 3]);

    // axis 3: digit = w >> 2
    for (int c = 0; c < 4; c++) bfly4(W[c], W[c + 4], W[c + 8], W[c + 12]);
}

// Fused rotate + row-wise INT8 quantize. x is bfloat16 [M,K].
// SUBG lanes per row; each lane owns G ConvRot groups held in registers.
template <int SUBG, int G>
static void launch(queue &q, const uint16_t *x, int8_t *qout, float *scale, size_t M,
                   size_t K) {
    const size_t n_groups = K / GS;
    const F16 c0 = h4_col(0), c1 = h4_col(1), c2 = h4_col(2), c3 = h4_col(3);
    const F16 d0 = h4_col_d1(0), d1 = h4_col_d1(1), d2 = h4_col_d1(2), d3 = h4_col_d1(3);

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M * SUBG), range<1>(SUBG)),
                       [=](nd_item<1> it) {
                           const size_t row = it.get_global_id(0) / SUBG;
                           const size_t lane = it.get_sub_group().get_local_linear_id();
                           const uint16_t *xr = x + row * K;

                           F16 W[G][NREG];
                           F16 amaxv(0.f);

                           for (int g = 0; g < G; ++g) {
                               const size_t grp = static_cast<size_t>(g) * SUBG + lane;
                               if (grp >= n_groups) continue;
                               const uint16_t *p = xr + grp * GS;

                               for (int w = 0; w < NREG; ++w) W[g][w] = load_reg(p + w * 16);
                               rotate_group(W[g], c0, c1, c2, c3, d0, d1, d2, d3);

                               for (int w = 0; w < NREG; ++w)
                                   amaxv = sycl::max(amaxv, sycl::fabs(W[g][w]));
                           }

                           // horizontal max of the lane accumulator, then sub-group max
                           F16 r0 = sycl::max(amaxv, amaxv.swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());
                           F16 r1 = sycl::max(r0, r0.swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>());
                           F16 r2 = sycl::max(r1, r1.swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>());
                           F16 r3 = sycl::max(r2, r2.swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>());
                           const float amax = r3[0];
                           const auto sg = it.get_sub_group();
                           float v = amax;
                           v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 8)));
                           v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 4)));
                           v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 2)));
                           v = sycl::fmax(v, sycl::select_from_group(sg, v, (uint32_t)(lane ^ 1)));
                           const float s = sycl::fmax(v / 127.0f, 1e-30f);
                           if (lane == 0) scale[row] = s;
                           const float inv = 1.0f / s;

                           for (int g = 0; g < G; ++g) {
                               const size_t grp = static_cast<size_t>(g) * SUBG + lane;
                               if (grp >= n_groups) continue;
                               int8_t *p = qout + row * K + grp * GS;
                               for (int w = 0; w < NREG; ++w) store_reg(p + w * 16, W[g][w], inv);
                           }
                       });
    }).wait();
}

}  // namespace xpuint8

// ---------------------------------------------------------------------------
// Reference + test/benchmark driver.
// ---------------------------------------------------------------------------
using namespace xpuint8;

static const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};

static float f32_from_bf16(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static void ref_rotate(const std::vector<float> &in, std::vector<float> &out, size_t M,
                       size_t K) {
    const int G = GS;
    const float nrm = 1.0f / std::sqrt(static_cast<float>(G));
    out.assign(in.size(), 0.f);
    for (size_t m = 0; m < M; m++)
        for (size_t g = 0; g < K / G; g++) {
            const float *src = in.data() + m * K + g * G;
            float *dst = out.data() + m * K + g * G;
            for (int i = 0; i < G; i++) {
                const int i0 = i & 3, i1 = (i >> 2) & 3, i2 = (i >> 4) & 3, i3 = (i >> 6) & 3;
                float acc = 0.f;
                for (int j = 0; j < G; j++) {
                    const int j0 = j & 3, j1 = (j >> 2) & 3, j2 = (j >> 4) & 3,
                              j3 = (j >> 6) & 3;
                    acc += src[j] * H4[i0][j0] * H4[i1][j1] * H4[i2][j2] * H4[i3][j3];
                }
                dst[i] = acc * nrm;
            }
        }
}

int main(int argc, char **argv) {
    const size_t M = argc > 1 ? strtoul(argv[1], nullptr, 10) : 4096;
    const size_t K = argc > 2 ? strtoul(argv[2], nullptr, 10) : 3072;
    const bool verify = argc > 3 ? (atoi(argv[3]) != 0) : true;

    queue q(gpu_selector_v);
    std::printf("%s  M=%zu K=%zu\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), M, K);

    std::vector<uint16_t> hx(M * K);
    std::vector<float> hxf(M * K);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < hx.size(); i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        float v = static_cast<float>(static_cast<int>((s >> 33) % 2001) - 1000) / 1000.f;
        v *= static_cast<float>((i % 7) + 1);
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u += (u >> 16) + 0x7FFFu;
        hx[i] = static_cast<uint16_t>(u >> 16);
        hxf[i] = f32_from_bf16(hx[i]);
    }

    uint16_t *x = malloc_shared<uint16_t>(M * K, q);
    int8_t *qo = malloc_shared<int8_t>(M * K, q);
    float *sc = malloc_shared<float>(M, q);
    for (size_t i = 0; i < hx.size(); i++) x[i] = hx[i];

    constexpr int SUBG = 16;
    const int gpl = static_cast<int>((K / GS + SUBG - 1) / SUBG);
    auto run = [&] {
        switch (gpl) {
        case 1: launch<SUBG, 1>(q, x, qo, sc, M, K); break;
        case 2: launch<SUBG, 2>(q, x, qo, sc, M, K); break;
        case 3: launch<SUBG, 3>(q, x, qo, sc, M, K); break;
        case 4: launch<SUBG, 4>(q, x, qo, sc, M, K); break;
        case 6: launch<SUBG, 6>(q, x, qo, sc, M, K); break;
        default: launch<SUBG, 8>(q, x, qo, sc, M, K); break;
        }
    };
    run();

    bool ok = true;
    if (verify) {
        std::vector<float> xr_ref;
        ref_rotate(hxf, xr_ref, M, K);
        std::vector<int8_t> hq(M * K);
        std::vector<float> hs(M);
        for (size_t i = 0; i < hq.size(); i++) hq[i] = qo[i];
        for (size_t i = 0; i < M; i++) hs[i] = sc[i];

        int scale_bad = 0;
        double worst = 0.0;
        for (size_t m = 0; m < M; m++) {
            float amax = 0.f;
            for (size_t k = 0; k < K; k++) amax = std::fmax(amax, std::fabs(xr_ref[m * K + k]));
            const float sr = std::fmax(amax / 127.0f, 1e-30f);
            if (std::fabs(sr - hs[m]) > sr * 1e-5f + 1e-30f) scale_bad++;
            for (size_t k = 0; k < K; k++) {
                float ref = std::rint(xr_ref[m * K + k] / sr);
                ref = std::fmin(127.f, std::fmax(-127.f, ref));
                worst = std::max(worst, (double)std::fabs(ref - (float)hq[m * K + k]));
            }
        }
        std::printf("scale mismatches      : %d / %zu\n", scale_bad, M);
        std::printf("max int8 deviation    : %.0f\n", worst);
        ok = (scale_bad == 0 && worst <= 1.0);
        std::printf("correctness           : %s\n", ok ? "PASS" : "FAIL");
    }

    const int iters = M >= 2048 ? 20 : 100;
    for (int i = 0; i < 3; i++) run();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run();
    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double gb = static_cast<double>(M) * K * 3.0 / 1e9;
    std::printf("time                  : %.4f ms  (%.0f GB/s effective)\n", ms,
                gb / (ms * 1e-3));

    sycl::free(x, q);
    sycl::free(qo, q);
    sycl::free(sc, q);
    return ok ? 0 : 1;
}