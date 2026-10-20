// Tiled ESIMD INT8 GEMM for Intel Arc (Xe2 / XMX DPAS).
//
// DPAS facts established by the probe in kernels/probe512b.cpp:
//   dpas<8,1,int,s8,s8,512,32>(B, A) yields 16 output columns, each a 32-element
//   int8 dot product accumulated into int32. All 32 A lanes are consumed in
//   order; A[i] pairs with B[ACTIVE_B[i]].
//   ACTIVE_B = even VNNI groups only (8 groups of 4 bytes, 128-byte stride).
//
// Usage: gemm_tiled [M N K] [ROWS]
//   ROWS = number of A rows computed per work item (B-load reuse factor).

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;

static const int ACTIVE_B[32] = {
    0,   1,   2,   3,   64,  65,  66,  67,  128, 129, 130, 131,
    192, 193, 194, 195, 256, 257, 258, 259, 320, 321, 322, 323,
    384, 385, 386, 387, 448, 449, 450, 451};

static constexpr int BN = 512;
static constexpr int AN = 32;
static constexpr int ON = 16;

static std::vector<signed char>
pack_weights(const std::vector<signed char> &w, int N, int K) {
    const int kchunks = K / AN;
    const int ntiles = N / ON;
    std::vector<signed char> p((size_t)N * K, 0);
    for (int nt = 0; nt < ntiles; ++nt)
        for (int kc = 0; kc < kchunks; ++kc) {
            signed char *blk = p.data() + ((size_t)nt * kchunks + kc) * BN;
            for (int i = 0; i < AN; ++i) {
                int k = kc * AN + i;
                for (int j = 0; j < ON; ++j)
                    blk[ACTIVE_B[i] + j * 4] = w[(size_t)(nt * ON + j) * K + k];
            }
        }
    return p;
}

template <int ROWS>
static void launch(queue &q, const signed char *a, const signed char *bp, int *c,
                   int M, int N, int K) {
    const int kchunks = K / AN;
    const size_t blk_stride = (size_t)kchunks * BN;
    q.submit([&](handler &h) {
        h.parallel_for(range<2>((M + ROWS - 1) / ROWS, N / ON), [=](id<2> idx) {
            int m0 = idx[0] * ROWS;
            int nt = idx[1];
            simd<signed char, BN> B(0);
            simd<int, ON> acc[ROWS];
            for (int r = 0; r < ROWS; r++)
                for (int j = 0; j < ON; j++) acc[r][j] = 0;

            const signed char *bptr = bp + nt * blk_stride;
            for (int kc = 0; kc < kchunks; ++kc) {
                const signed char *bblk = bptr + (size_t)kc * BN;
                // one vectorised 512-byte load, not 512 scalar byte loads
                std::memcpy(&B, bblk, BN);
                for (int r = 0; r < ROWS; r++) {
                    simd<signed char, AN> A(0);
                    const signed char *arow = a + (size_t)(m0 + r) * K + kc * AN;
                    std::memcpy(&A, arow, AN);
                    simd<int, ON> t = dpas<8, 1, int, signed char, signed char,
                                          dpas_argument_type::s8,
                                          dpas_argument_type::s8, BN, AN>(B, A);
                    for (int j = 0; j < ON; j++) acc[r][j] = acc[r][j] + t[j];
                }
            }
            for (int r = 0; r < ROWS; r++) {
                int m = m0 + r;
                if (m >= M) break;
                int *crow = c + (size_t)m * N + nt * ON;
                for (int j = 0; j < ON; j++) crow[j] = acc[r][j];
            }
        });
    }).wait();
}

int main(int argc, char **argv) {
    int M = argc > 1 ? atoi(argv[1]) : 256;
    int N = argc > 2 ? atoi(argv[2]) : 256;
    int K = argc > 3 ? atoi(argv[3]) : 256;
    int ROWS = argc > 4 ? atoi(argv[4]) : 4;

    queue q(gpu_selector_v);
    std::printf("%s  M=%d N=%d K=%d ROWS=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                M, N, K, ROWS);

    std::vector<signed char> ha((size_t)M * K), hb((size_t)N * K);
    for (size_t i = 0; i < ha.size(); i++)
        ha[i] = static_cast<signed char>((i * 37 + 1) % 255 - 127);
    for (size_t i = 0; i < hb.size(); i++)
        hb[i] = static_cast<signed char>((i * 19 + 7) % 255 - 127);
    std::vector<signed char> hbp = pack_weights(hb, N, K);

    signed char *a = malloc_shared<signed char>((size_t)M * K, q);
    signed char *bp = malloc_shared<signed char>((size_t)N * K, q);
    int *c = malloc_shared<int>((size_t)M * N, q);
    for (size_t i = 0; i < ha.size(); i++) a[i] = ha[i];
    for (size_t i = 0; i < hbp.size(); i++) bp[i] = hbp[i];

    auto run = [&] {
        switch (ROWS) {
        case 1: launch<1>(q, a, bp, c, M, N, K); break;
        case 2: launch<2>(q, a, bp, c, M, N, K); break;
        case 4: launch<4>(q, a, bp, c, M, N, K); break;
        case 8: launch<8>(q, a, bp, c, M, N, K); break;
        default: launch<16>(q, a, bp, c, M, N, K); break;
        }
    };

    run();

    // Verify a sample of outputs against the CPU reference.
    bool ok = true;
    int checked = 0;
    for (int m = 0; m < M && ok && checked < 64; m += (M / 8 + 1))
        for (int n = 0; n < N && checked < 64; n += (N / 8 + 1)) {
            int ref = 0;
            for (int k = 0; k < K; k++)
                ref += (int)ha[(size_t)m * K + k] * (int)hb[(size_t)n * K + k];
            if (ref != c[(size_t)m * N + n]) {
                std::printf("MISMATCH m=%d n=%d ref=%d got=%d\n", m, n, ref,
                            c[(size_t)m * N + n]);
                ok = false;
                break;
            }
            checked++;
        }
    std::printf("correctness: %s (%d samples)\n", ok ? "PASS" : "FAIL", checked);

    int iters = M > 1024 ? 10 : 50;
    for (int r = 0; r < 3; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; it++) run();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double tops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        std::printf("  %.3f ms   %.2f TOPS\n", ms, tops);
    }

    sycl::free(a, q);
    sycl::free(bp, q);
    sycl::free(c, q);
    return ok ? 0 : 1;
}