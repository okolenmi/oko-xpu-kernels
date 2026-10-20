// INT8 GEMM via ESIMD DPAS, tuned for Intel Arc (Xe2).
//
//   C[M,N] = A[M,K] * B[N,K]^T     int8 x int8 -> int32
//
// DPAS facts (established by kernels/probe512b.cpp):
//   dpas<8, 1, int, s8, s8, 512, 32>(B, A) computes 16 output columns, each a
//   32-element int8 dot product. All 32 lanes of A are consumed in order:
//   A[i] pairs with B[ACTIVE_B[i]], where ACTIVE_B are the *even* VNNI groups
//   (group stride 128 bytes, 4 bytes per group).
//
// So B is pre-packed offline into that layout and the inner loop is a flat
// 512-byte load per (32-K chunk, 16-N column tile).
//
// This ESIMD build has no arithmetic operators on simd<> (only dpas), so the
// accumulators are sycl::vec<int32_t,16> and the DPAS operands/results are
// bridged with memcpy, which the compiler turns into pure register moves.
//
// Usage: gemm_v3 [M N K] [ROWS]

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

static const int ACTIVE_B[32] = {
    0,   1,   2,   3,   64,  65,  66,  67,  128, 129, 130, 131,
    192, 193, 194, 195, 256, 257, 258, 259, 320, 321, 322, 323,
    384, 385, 386, 387, 448, 449, 450, 451};

constexpr int BN = 512;  // 16 cols x 32 int8, DPAS B operand
constexpr int AN = 32;   // 32-element dot product
constexpr int ON = 16;   // output columns per DPAS

using ACC = sycl::vec<int32_t, ON>;

static std::vector<int8_t> pack_weights(const std::vector<int8_t> &w, int N, int K) {
    const int kchunks = K / AN, ntiles = N / ON;
    std::vector<int8_t> p((size_t)N * K, 0);
    for (int nt = 0; nt < ntiles; ++nt)
        for (int kc = 0; kc < kchunks; ++kc) {
            int8_t *blk = p.data() + ((size_t)nt * kchunks + kc) * BN;
            for (int i = 0; i < AN; ++i) {
                int k = kc * AN + i;
                for (int j = 0; j < ON; ++j) blk[ACTIVE_B[i] + j * 4] = w[(size_t)(nt * ON + j) * K + k];
            }
        }
    return p;
}

template <int ROWS>
static void launch(queue &q, const int8_t *a, const int8_t *bp, int32_t *c, int M, int N,
                   int K) {
    const int kchunks = K / AN;
    const size_t blk_stride = (size_t)kchunks * BN;

    q.submit([&](handler &h) {
        h.parallel_for(range<2>((M + ROWS - 1) / ROWS, N / ON), [=](id<2> idx) {
            const int m0 = idx[0] * ROWS;
            const int nt = idx[1];
            const int8_t *bptr = bp + nt * blk_stride;

            ACC acc[ROWS];
            for (int r = 0; r < ROWS; r++) acc[r] = ACC(0);

            for (int kc = 0; kc < kchunks; ++kc) {
                simd<signed char, BN> B;
                std::memcpy(&B, bptr + (size_t)kc * BN, BN);
                for (int r = 0; r < ROWS; ++r) {
                    simd<signed char, AN> A;
                    std::memcpy(&A, a + (size_t)(m0 + r) * K + kc * AN, AN);
                    simd<int, ON> d = dpas<8, 1, int, signed char, signed char,
                                             dpas_argument_type::s8,
                                             dpas_argument_type::s8, BN, AN>(B, A);
                    ACC t;
                    std::memcpy(&t, &d, sizeof(t));
                    acc[r] = acc[r] + t;  // one vector add, not 16 scalar adds
                }
            }
            for (int r = 0; r < ROWS; ++r) {
                if (m0 + r >= M) break;
                std::memcpy(c + (size_t)(m0 + r) * N + nt * ON, &acc[r], sizeof(ACC));
            }
        });
    }).wait();
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 3072;
    int K = argc > 3 ? atoi(argv[3]) : 3072;
    int ROWS = argc > 4 ? atoi(argv[4]) : 16;

    queue q;
    std::printf("%s  M=%d N=%d K=%d ROWS=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), M, N, K, ROWS);

    std::vector<int8_t> ha((size_t)M * K), hb((size_t)N * K);
    for (size_t i = 0; i < ha.size(); i++) ha[i] = (int8_t)((i * 37 + 1) % 255 - 127);
    for (size_t i = 0; i < hb.size(); i++) hb[i] = (int8_t)((i * 19 + 7) % 255 - 127);
    std::vector<int8_t> hbp = pack_weights(hb, N, K);

    int8_t *a = malloc_shared<int8_t>((size_t)M * K, q);
    int8_t *bp = malloc_shared<int8_t>((size_t)N * K, q);
    int32_t *c = malloc_shared<int32_t>((size_t)M * N, q);
    std::memcpy(a, ha.data(), ha.size());
    std::memcpy(bp, hbp.data(), hbp.size());

    auto run = [&] {
        switch (ROWS) {
        case 2: launch<2>(q, a, bp, c, M, N, K); break;
        case 4: launch<4>(q, a, bp, c, M, N, K); break;
        case 6: launch<6>(q, a, bp, c, M, N, K); break;
        case 10: launch<10>(q, a, bp, c, M, N, K); break;
        case 12: launch<12>(q, a, bp, c, M, N, K); break;
        case 8: launch<8>(q, a, bp, c, M, N, K); break;
        case 16: launch<16>(q, a, bp, c, M, N, K); break;
        default: launch<32>(q, a, bp, c, M, N, K); break;
        }
    };
    run();

    // verify a sample
    bool ok = true;
    int checked = 0;
    for (int m = 0; m < M && ok && checked < 48; m += (M / 6 + 1))
        for (int n = 0; n < N && checked < 48; n += (N / 6 + 1)) {
            int32_t ref = 0;
            for (int k = 0; k < K; k++) ref += (int)ha[(size_t)m * K + k] * (int)hb[(size_t)n * K + k];
            if (ref != c[(size_t)m * N + n]) {
                std::printf("MISMATCH m=%d n=%d ref=%d got=%d\n", m, n, ref, c[(size_t)m * N + n]);
                ok = false;
                break;
            }
            checked++;
        }
    std::printf("correctness: %s (%d samples)\n", ok ? "PASS" : "FAIL", checked);

    int iters = (size_t)M * N > (1 << 24) ? 10 : 30;
    for (int i = 0; i < 3; i++) run();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) run();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("  %.4f ms   %.2f TOPS\n", ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12);

    sycl::free(a, q); sycl::free(bp, q); sycl::free(c, q);
    return ok ? 0 : 1;
}