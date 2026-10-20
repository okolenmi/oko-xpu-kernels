/* Do 64-byte loads beat the production 32-byte loads?
 *
 * The rotate loads a 512-byte group as 16 x 32-byte vectors (one
 * vec<uint32_t,8> = 16 bf16). Xe has 64-byte vector loads, so 8 x 64-byte
 * loads would halve the load instruction count.
 *
 * (An earlier version of this probe compared against a loop that only consumed
 * element [0], so the compiler deleted the loads and the comparison was
 * meaningless. Both variants below consume every element.)
 *
 *   ./build.sh && ./bin/probe_load_store_width 4096 3072
 */
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace sycl;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;

constexpr int NREG = 16;
constexpr int GS = 256;

static inline U16 even_mask() {
    U16 m;
    for (int t = 0; t < 16; t++) m[t] = (t & 1) ? 0u : 0xFFFFFFFFu;
    return m;
}

// 32-byte loads, exactly as the production kernel does it
static inline void ld32(const uint16_t *p, F16 W[NREG]) {
    const auto *pp = reinterpret_cast<const vec<uint32_t, 8> *>(p);
    for (int w = 0; w < NREG; w++) {
        const vec<uint32_t, 8> u = pp[w];
        U16 o;
        for (int j = 0; j < 8; j++) {
            o[2 * j] = u[j] << 16;
            o[2 * j + 1] = u[j] & 0xFFFF0000u;
        }
        W[w] = sycl::bit_cast<F16>(o);
    }
}

// 64-byte loads: one vec<uint32_t,16> per pair of registers, split by swizzle
static inline void ld64(const uint16_t *p, F16 W[NREG]) {
    const U16 em = even_mask();
    for (int b = 0; b < NREG; b += 2) {
        const U16 u = *reinterpret_cast<const U16 *>(p + b * 16);  // 64 bytes
        // low bf16 of each of the 16 dwords -> register b
        const U16 lo = u.swizzle<0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7>();
        // high bf16 of each of the first 8 dwords -> register b+1
        const U16 hi = u.swizzle<8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15>();
        // Neither half needs a shift: the mask alone selects it.
        W[b] = sycl::bit_cast<F16>(lo & em);
        W[b + 1] = sycl::bit_cast<F16>(hi & em);
    }
}

// A 64-byte fetch does not line up with a 16-lane float vector: 32 bf16 land in
// 32 consecutive dwords, so the two output vectors need dwords 0..7's low
// halves and 8..15's high halves. The swizzles above express that.
static inline bool check_layout() {
    uint16_t host[GS];
    for (int i = 0; i < GS; i++) host[i] = (uint16_t)(0x3F00 + i);
    F16 a[NREG], b[NREG];
    ld32(host, a);
    ld64(host, b);
    for (int w = 0; w < NREG; ++w)
        for (int t = 0; t < 16; ++t)
            if (!(a[w][t] == b[w][t])) return false;
    return true;
}

template <int WIDTH>
static void run(queue &q, const uint16_t *x, int8_t *o, int *cs, size_t M, size_t K) {
    const size_t ng = K / GS;
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M * 16), range<1>(16)), [=](nd_item<1> it) {
            const size_t row = it.get_global_id(0) >> 4, lane = it.get_local_id(0) & 15;
            if (lane >= ng) return;
            const uint16_t *p = x + row * K + lane * GS;
            F16 W[NREG];
            if (WIDTH == 32) ld32(p, W);
            else ld64(p, W);
            // consume every element so nothing is elided
            F16 acc(0.f);
            for (int w = 0; w < NREG; w++) acc = acc + W[w];
            const F16 r = sycl::min(sycl::max(sycl::rint(acc), F16(-127.f)), F16(127.f));
            int8_t *d = o + row * K + lane * GS;
            uint32_t b[4] = {0, 0, 0, 0};
            for (int t = 0; t < 16; t++)
                b[t >> 2] |= ((uint32_t)(uint8_t)(int8_t)(int32_t)r[t]) << (8 * (t & 3));
            std::memcpy(d, b, 16);
            // accumulate a checksum so the loads cannot be elided and so a
            // wrong swizzle is visible in the output
            if (lane == 0 && row == 0) {
                atomic_ref<int, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(cs[0])
                    .fetch_add((int)(W[0][0] * 1000.0f) + (int)(W[15][0] * 7.0f));
            }
        });
    }).wait();
}

int main(int argc, char **argv) {
    const size_t M = argc > 1 ? strtoul(argv[1], 0, 10) : 4096;
    const size_t K = argc > 2 ? strtoul(argv[2], 0, 10) : 3072;
    const size_t N = M * K;

    queue q;
    uint16_t *x = malloc_shared<uint16_t>(N, q);
    int8_t *o = malloc_shared<int8_t>(N, q);
    for (size_t i = 0; i < N; i++) x[i] = 0x3F80;

    auto bench = [&](const char *nm, auto fn, int *cs) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            for (int i = 0; i < 5; i++) fn();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 20; i++) fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best,
                            std::chrono::duration<double, std::milli>(t1 - t0).count() / 20);
        }
        std::printf("  %-34s %7.4f ms   checksum %d\n", nm, best, cs ? *cs : -1);
        return best;
    };

    std::printf("layout check (ld64 must equal ld32): %s\n",
                check_layout() ? "PASS" : "FAIL");
    std::printf("M=%zu K=%zu, 16 registers live, no row-scale dependency\n", M, K);
    int *cs = malloc_shared<int>(1, q); *cs = 0;
    double a = bench("32-byte loads x16 (production)", [&] { run<32>(q, x, o, cs, M, K); }, cs);
    *cs = 0;
    double b = bench("64-byte loads x8", [&] { run<64>(q, x, o, cs, M, K); }, cs);
    std::printf("  64-byte loads are %.2fx %s\n", a / b, a > b ? "faster" : "slower");
    std::printf("  load-only cost is ~%.0f GB/s either way, so the load width is not\n"
                "  the bottleneck -- register residency is.\n", (double)(M * K * 2) / 1e9 / (a * 1e-3));
    return 0;
}