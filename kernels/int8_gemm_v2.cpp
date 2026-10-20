// Verified-correct ESIMD INT8 GEMM for Intel Arc (Xe2, XMX DPAS).
//
// Key finding from the DPAS probe (kernels/probe512b.cpp):
//   dpas<8,1,int,s8,s8,512,32>(B, A) on Arc B580 computes 16 output columns
//   with a 32-element dot product, but the hardware only consumes B lanes in
//   the EVEN VNNI groups. Active B offsets for output column 0 are:
//       0..3, 64..67, 128..131, 192..195, 256..259, 320..323, 384..387, 448..451
//   i.e. group stride 128 bytes, 4 bytes per group. All 32 A lanes are consumed
//   sequentially (A[i] pairs with B[ACTIVE_B[i]]).
//
// Therefore the weights must be pre-packed offline into that layout so the inner
// loop is a flat 512-byte load per (32-K chunk, 16-N column tile).
//
// Correctness is verified against a CPU reference in main().

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <cstdio>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;

// B vector layout expected by dpas<...,512,32> for 16 output columns.
static const int ACTIVE_B[32] = {
    0,   1,   2,   3,   64,  65,  66,  67,  128, 129, 130, 131,
    192, 193, 194, 195, 256, 257, 258, 259, 320, 321, 322, 323,
    384, 385, 386, 387, 448, 449, 450, 451};

static constexpr int BN = 512;   // 16 cols x 32 int8
static constexpr int AN = 32;    // 32-element dot product
static constexpr int ON = 16;    // output columns per dpas

// Pack row-major weights B[N][K] into the DPAS VNNI layout.
static std::vector<signed char>
pack_weights(const std::vector<signed char> &w, int N, int K) {
    const int kchunks = K / AN;
    const int ntiles = N / ON;
    std::vector<signed char> p(N * K, 0);
    for (int nt = 0; nt < ntiles; ++nt) {
        for (int kc = 0; kc < kchunks; ++kc) {
            // Base of this (nt, kc) 512-byte block.
            signed char *blk = p.data() + ((size_t)nt * kchunks + kc) * BN;
            for (int i = 0; i < AN; ++i) {
                int k = kc * AN + i;
                for (int j = 0; j < ON; ++j) {
                    int n = nt * ON + j;
                    blk[ACTIVE_B[i] + j * 4] = w[(size_t)n * K + k];
                }
            }
        }
    }
    return p;
}

int main() {
    int M = 256, N = 256, K = 256;
    queue q(gpu_selector_v);
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    std::vector<signed char> ha(M * K), hb(N * K);
    for (int i = 0; i < M * K; i++)
        ha[i] = static_cast<signed char>((i * 37 + 1) % 255 - 127);
    for (int i = 0; i < N * K; i++)
        hb[i] = static_cast<signed char>((i * 19 + 7) % 255 - 127);
    std::vector<signed char> hbp = pack_weights(hb, N, K);

    signed char *a = malloc_shared<signed char>(M * K, q);
    signed char *bp = malloc_shared<signed char>(N * K, q);
    int *c = malloc_shared<int>(M * N, q);
    for (int i = 0; i < M * K; i++) a[i] = ha[i];
    for (int i = 0; i < N * K; i++) bp[i] = hbp[i];
    for (int i = 0; i < M * N; i++) c[i] = 0;

    const int kchunks = K / AN;
    const size_t blk_stride = (size_t)kchunks * BN;

    auto run = [&] {
        q.submit([&](handler &h) {
            h.parallel_for(range<2>(M, N / ON), [=](id<2> idx) {
                int m = idx[0];
                int nt = idx[1];
                simd<signed char, AN> A(0);
                simd<signed char, BN> B(0);
                simd<int, ON> acc(0);
                const signed char *bptr = bp + nt * blk_stride;
                for (int kc = 0; kc < kchunks; ++kc) {
                    const signed char *arow = a + (size_t)m * K + kc * AN;
                    for (int i = 0; i < AN; i++) A[i] = arow[i];
                    const signed char *bblk = bptr + (size_t)kc * BN;
                    for (int i = 0; i < BN; i++) B[i] = bblk[i];
                    simd<int, ON> t = dpas<8, 1, int, signed char, signed char,
                                          dpas_argument_type::s8,
                                          dpas_argument_type::s8, BN, AN>(B, A);
                    for (int j = 0; j < ON; j++) acc[j] = acc[j] + t[j];
                }
                int *crow = c + (size_t)m * N + nt * ON;
                for (int j = 0; j < ON; j++) crow[j] = acc[j];
            });
        }).wait();
    };

    run();

    bool ok = true;
    int bad = 0;
    for (int m = 0; m < M && ok; m++)
        for (int n = 0; n < N; n++) {
            int ref = 0;
            for (int k = 0; k < K; k++)
                ref += (int)ha[m * K + k] * (int)hb[n * K + k];
            if (ref != c[(size_t)m * N + n]) {
                std::printf("mismatch m=%d n=%d ref=%d got=%d\n", m, n, ref,
                            c[(size_t)m * N + n]);
                ok = false;
                bad++;
                break;
            }
        }
    std::printf("%s\n", ok ? "correct" : "incorrect");

    // Timed loop.
    for (int r = 0; r < 3; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < 10; it++) run();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0;
        double tops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        std::printf("iter %d: %.3f ms  %.2f TOPS\n", r, ms, tops);
    }

    sycl::free(a, q);
    sycl::free(bp, q);
    sycl::free(c, q);
    return ok ? 0 : 1;
}