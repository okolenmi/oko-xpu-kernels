/*
 * Where the weight dequantizer's time goes, and whether a cheaper form of its arithmetic is bit-identical.
 *
 * The production kernel (src/xpuint8.cpp, launch_dequant_weight) reads int8, multiplies by the row scale, rounds to bf16,
 * undoes the ConvRot rotation in 16 registers per lane and writes bf16: 3 bytes per parameter, at ~60-150 GB/s on a card
 * whose copy ceiling is ~400.  Each MODE below removes or replaces one piece, so the cost of each can be read off:
 *
 *   0  production
 *   1  without the rotation (load, dequantize, store)
 *   2  with the rotation, the int8 -> float conversion only (no scale, no bf16 rounding)
 *   3  neither the rotation nor the scale and rounding
 *   4  the production arithmetic with the dequantization written on whole vectors (vec<int8>::convert, vector bf16 rounding)
 *   5  as 4, and the bf16 store as one vector convert
 *   6  a coalesced elementwise dequantizer (16 contiguous parameters per work-item, no rotation): the floor of any form
 *   7  as 6 without the scale (the pure int8 -> bf16 widening)
 *
 * Modes 4 and 5 must be bit-identical to 0 (printed); the rest compute something else on purpose.
 *
 *   ./build.sh && ./bin/probe_dequant_weight 1280 5120      # N K  (default 1280 5120)
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;
using BF16 = vec<BF, 16>;

constexpr int GS = 256, NREG = 16, SUBG = 16;
static const float H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
static inline F16 h4_col(int c, bool d1) { F16 v; for (int t = 0; t < 16; t++) v[t] = 0.5f * H4[d1 ? ((t >> 2) & 3) : (t & 3)][c]; return v; }
static inline void bfly4(F16 &a0, F16 &a1, F16 &a2, F16 &a3) {
    const F16 p = a0 + a1, m = a0 - a1, q = a2 + a3, n = a2 - a3;
    a0 = (p + n) * 0.5f; a1 = (p - n) * 0.5f; a2 = (m + q) * 0.5f; a3 = (q - m) * 0.5f;
}
static inline F16 inreg(F16 v, const F16 &c0, const F16 &c1, const F16 &c2, const F16 &c3,
                        const F16 &d0, const F16 &d1, const F16 &d2, const F16 &d3) {
    F16 t = c0 * v.swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>();
    t = t + c1 * v.swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>();
    t = t + c2 * v.swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>();
    t = t + c3 * v.swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
    F16 u = d0 * t.swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>();
    u = u + d1 * t.swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>();
    u = u + d2 * t.swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>();
    u = u + d3 * t.swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
    return u;
}

// the production load: scalar unpack, fp32 multiply, bf16 rounding, back to fp32
static inline F16 load_prod(const int8_t *p, float scale) {
    const vec<uint32_t, 4> u = reinterpret_cast<const vec<uint32_t, 4> *>(p)[0];
    F16 v;
    for (int t = 0; t < 16; t++) {
        const uint32_t byte = (u[t >> 2] >> (8 * (t & 3))) & 0xFFu;
        const int32_t s = static_cast<int32_t>(static_cast<int8_t>(byte));
        v[t] = static_cast<float>(BF(static_cast<float>(s) * scale));
    }
    return v;
}
static inline F16 load_plain(const int8_t *p) {  // widening only
    const vec<int8_t, 16> b = reinterpret_cast<const vec<int8_t, 16> *>(p)[0];
    return b.convert<float>();
}
static inline F16 load_vec(const int8_t *p, float scale) {  // the production arithmetic on whole vectors
    F16 v = load_plain(p) * scale;
    return v.template convert<BF>().template convert<float>();
}
static inline void store_scalar(BF *d, const F16 &v) {
    BF16 r;
    for (int t = 0; t < 16; t++) r[t] = BF(v[t]);
    reinterpret_cast<BF16 *>(d)[0] = r;
}
static inline void store_vec(BF *d, const F16 &v) { reinterpret_cast<BF16 *>(d)[0] = v.template convert<BF>(); }

template <int MODE>
static void run(queue &q, const int8_t *w8, const float *ws, BF *out, size_t N, size_t K) {
    const size_t ng = K / GS, gpl = (ng + SUBG - 1) / SUBG;
    const F16 c0 = h4_col(0, 0), c1 = h4_col(1, 0), c2 = h4_col(2, 0), c3 = h4_col(3, 0);
    const F16 d0 = h4_col(0, 1), d1 = h4_col(1, 1), d2 = h4_col(2, 1), d3 = h4_col(3, 1);
    if constexpr (MODE >= 6) {
        const size_t items = N * K / 16;
        q.submit([&](handler &h) {
            h.parallel_for(range<1>(items), [=](id<1> i) {
                const size_t at = i[0] * 16, row = at / K;
                F16 v = load_plain(w8 + at);
                if (MODE == 6) v = v * ws[row];
                store_vec(out + at, v);
            });
        }).wait();
        return;
    }
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(N * SUBG), range<1>(SUBG)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) >> 4, lane = it.get_local_id(0) & 15;
            const int8_t *src = w8 + row * K;
            BF *dst = out + row * K;
            const float scale = ws[row];
            for (size_t g = 0; g < gpl; ++g) {
                const size_t grp = g * SUBG + lane;
                if (grp >= ng) continue;
                const int8_t *p = src + grp * GS;
                F16 W[NREG];
                for (int b = 0; b < NREG; b += 4) {
                    for (int j = 0; j < 4; j++) {
                        const int8_t *s = p + (b + j) * 16;
                        if (MODE == 0 || MODE == 1) W[b + j] = load_prod(s, scale);
                        else if (MODE == 2 || MODE == 3) W[b + j] = load_plain(s);
                        else W[b + j] = load_vec(s, scale);
                    }
                    if (MODE != 1 && MODE != 3) {
                        for (int j = 0; j < 4; j++) W[b + j] = inreg(W[b + j], c0, c1, c2, c3, d0, d1, d2, d3);
                        bfly4(W[b], W[b + 1], W[b + 2], W[b + 3]);
                    }
                }
                if (MODE != 1 && MODE != 3) for (int c = 0; c < 4; c++) bfly4(W[c], W[c + 4], W[c + 8], W[c + 12]);
                BF *d = dst + grp * GS;
                for (int w = 0; w < NREG; w++) {
                    if (MODE == 5) store_vec(d + w * 16, W[w]);
                    else store_scalar(d + w * 16, W[w]);
                }
            }
        });
    }).wait();
}

int main(int argc, char **argv) {
    queue q;
    const size_t N = argc > 1 ? strtoul(argv[1], 0, 10) : 1280, K = argc > 2 ? strtoul(argv[2], 0, 10) : 5120;
    std::printf("device: %s   N=%zu K=%zu (%.1f MB int8 in, %.1f MB bf16 out)\n\n", q.get_device().get_info<info::device::name>().c_str(),
                N, K, N * K / 1e6, N * K * 2 / 1e6);
    int8_t *w8 = malloc_shared<int8_t>(N * K, q);
    float *ws = malloc_shared<float>(N, q);
    BF *out = malloc_shared<BF>(N * K, q);
    BF *ref = malloc_shared<BF>(N * K, q);
    for (size_t i = 0; i < N * K; i++) w8[i] = static_cast<int8_t>((i * 2654435761u >> 24) - 128);
    for (size_t i = 0; i < N; i++) ws[i] = 0.001f + 0.00003f * (i % 97);
    auto bench = [&](const char *name, auto fn) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < 3; i++) fn();
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 10; i++) fn();
            const auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count() / 10);
        }
        std::printf("%-58s %8.4f ms %6.0f GB/s\n", name, best, N * K * 3 / (best / 1000) / 1e9);
    };
    run<0>(q, w8, ws, ref, N, K);
    auto same = [&] { return std::memcmp(out, ref, N * K * sizeof(BF)) == 0 ? "bit-identical to 0" : "DIFFERENT from 0"; };
    bench("0 production", [&] { run<0>(q, w8, ws, out, N, K); });
    bench("1 without the rotation", [&] { run<1>(q, w8, ws, out, N, K); });
    bench("2 rotation, widening only", [&] { run<2>(q, w8, ws, out, N, K); });
    bench("3 neither", [&] { run<3>(q, w8, ws, out, N, K); });
    bench("4 vector dequantization", [&] { run<4>(q, w8, ws, out, N, K); });
    std::printf("      -> %s\n", same());
    bench("5 vector dequantization and vector store", [&] { run<5>(q, w8, ws, out, N, K); });
    std::printf("      -> %s\n", same());
    bench("6 coalesced elementwise (the floor of any form)", [&] { run<6>(q, w8, ws, out, N, K); });
    bench("7 coalesced widening only", [&] { run<7>(q, w8, ws, out, N, K); });
    return 0;
}
