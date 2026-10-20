// INT8 GEMM v4: DPAS with RepeatCount and the accumulating form.
//
// v3 used  dpas<8,1,...>(B,A)  + a separate vec add per row, i.e. per k-chunk:
//     8 x 512B B-loads + 8 x 32B A-loads + 8 DPAS + 8 vector adds  = 32
// instructions for 8 units of work (1 unit = 1 row x 16 cols x 32 K).
//
// ESIMD also exposes:
//     dpas<8, R, T, CT, BT, AT, BP, AP, N, BN, AN>(C, B, A)  ==  C + A x B
// which consumes R*32 bytes of A (R rows) against ONE B operand and folds the
// accumulate into the DPAS itself. AN must be exactly R*32 bytes, so A has to
// be tiled as [m/R][kchunk][R][32] -- 256 contiguous bytes for R=8.
//
//     R=8: 8 B-loads + 4 A-loads + 1 DPAS = 13 instructions per 8 units
//          -> 2.5x fewer instructions per unit than v3.
//
// Usage: gemm_v4 [M N K] [R]
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;

static const int ACTIVE_B[32] = {0,   1,   2,   3,   64,  65,  66,  67,
                                 128, 129, 130, 131, 192, 193, 194, 195,
                                 256, 257, 258, 259, 320, 321, 322, 323,
                                 384, 385, 386, 387, 448, 449, 450, 451};

constexpr int BN = 512;   // 16 cols x 32 int8 (VNNI encoded)
constexpr int AN = 32;    // 32-element dot product
constexpr int ON = 16;    // output columns per DPAS

// B: [ntile][kchunk][512]
static std::vector<int8_t> pack_b(const std::vector<int8_t> &w, int N, int K) {
    const int kchunks = K / AN, ntiles = N / ON;
    std::vector<int8_t> p((size_t)N * K, 0);
    for (int nt = 0; nt < ntiles; ++nt)
        for (int kc = 0; kc < kchunks; ++kc) {
            int8_t *blk = p.data() + ((size_t)nt * kchunks + kc) * BN;
            for (int i = 0; i < AN; ++i)
                for (int j = 0; j < ON; ++j)
                    blk[ACTIVE_B[i] + j * 4] = w[(size_t)(nt * ON + j) * K + kc * AN + i];
        }
    return p;
}

// A row-major [M][K] -> tiled [M/R][K/32][R][32]
static std::vector<int8_t> tile_a(const std::vector<int8_t> &a, int M, int K, int R) {
    std::vector<int8_t> t((size_t)M * K);
    const int nkc = K / AN;
    for (int mt = 0; mt < M / R; ++mt)
        for (int kc = 0; kc < nkc; ++kc)
            for (int r = 0; r < R; ++r)
                std::memcpy(t.data() + (((size_t)mt * nkc + kc) * R + r) * AN,
                            a.data() + ((size_t)(mt * R + r) * K + kc * AN), AN);
    return t;
}

template <int R>
static void launch(queue &q, const int8_t *at, const int8_t *bp, int32_t *c, int M, int N,
                   int K) {
    const int nkc = K / AN;
    const int ntiles = N / ON;
    const int mtiles = M / R;

    q.submit([&](handler &h) {
        h.parallel_for(range<2>(mtiles, ntiles), [=](id<2> idx) {
            const int mt = idx[0], nt = idx[1];
            const int8_t *ap = at + (size_t)mt * nkc * R * AN;
            const int8_t *bptr = bp + (size_t)nt * nkc * BN;

            simd<int, R * ON> acc((int)0);
            for (int kc = 0; kc < nkc; ++kc) {
                simd<signed char, BN> B;
                std::memcpy(&B, bptr + (size_t)kc * BN, BN);
                simd<signed char, R * AN> A;
                std::memcpy(&A, ap + (size_t)kc * R * AN, R * AN);
                acc = dpas<8, R, int, int, signed char, signed char,
                          dpas_argument_type::s8, dpas_argument_type::s8,
                          R * ON, BN, R * AN>(acc, B, A);
            }
            int32_t tmp[R * ON];
            std::memcpy(tmp, &acc, sizeof(tmp));
            int32_t *cp = c + (size_t)mt * R * N + nt * ON;
            for (int r = 0; r < R; ++r)
                std::memcpy(cp + (size_t)r * N, tmp + r * ON, ON * sizeof(int32_t));
        });
    }).wait();
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 3072;
    int K = argc > 3 ? atoi(argv[3]) : 3072;
    int RR = argc > 4 ? atoi(argv[4]) : 8;

    queue q;
    std::printf("%s  M=%d N=%d K=%d R=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), M, N, K, RR);

    std::vector<int8_t> ha((size_t)M * K), hb((size_t)N * K);
    for (size_t i = 0; i < ha.size(); i++) ha[i] = (int8_t)((i * 37 + 1) % 255 - 127);
    for (size_t i = 0; i < hb.size(); i++) hb[i] = (int8_t)((i * 19 + 7) % 255 - 127);
    std::vector<int8_t> hbp = pack_b(hb, N, K);

    int8_t *a = malloc_shared<int8_t>((size_t)M * K, q);
    int8_t *bp = malloc_shared<int8_t>((size_t)N * K, q);
    int32_t *c = malloc_shared<int32_t>((size_t)M * N, q);
    std::memcpy(bp, hbp.data(), hbp.size());

    // upload once per R, then time only the kernel
    auto verify_and_time = [&](const std::vector<int8_t> &ta) {
        auto time_only = [&] {
            switch (RR) {
            case 1: launch<1>(q, a, bp, c, M, N, K); break;
            case 2: launch<2>(q, a, bp, c, M, N, K); break;
            case 4: launch<4>(q, a, bp, c, M, N, K); break;
            case 8: launch<8>(q, a, bp, c, M, N, K); break;
            }
        };
        sycl::queue{}.memcpy(a, ta.data(), ta.size()).wait();
        time_only();
        int checked = 0;
        bool ok = true;
        for (int m = 0; m < M && checked < 32 && ok; m += (M / 5 + 1))
            for (int n = 0; n < N && checked < 32 && ok; n += (N / 5 + 1)) {
                int32_t ref = 0;
                for (int k = 0; k < K; k++) ref += (int)ha[(size_t)m * K + k] * (int)hb[(size_t)n * K + k];
                if (ref != c[(size_t)m * N + n]) {
                    std::printf("  R=%2d MISMATCH m=%d n=%d ref=%d got=%d\n", RR, m, n, ref, c[(size_t)m * N + n]);
                    ok = false; break;
                }
                checked++;
            }
        int iters = (size_t)M * N > (1 << 24) ? 10 : 30;
        for (int i = 0; i < 3; i++) time_only();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) time_only();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        std::printf("  R=%d  %s  %.4f ms   %.2f TOPS   (%.0f%% of 344 DPAS peak)\n", RR,
                    ok ? "PASS" : "FAIL", ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12,
                    100 * (2.0 * M * N * K / (ms * 1e-3) / 1e12) / 344.0);
        return ok;
    };

    bool okall = true;
    for (int R : {1, 2, 4, 8}) { RR = R; okall &= verify_and_time(tile_a(ha, M, K, R)); }
    return okall ? 0 : 1;
}