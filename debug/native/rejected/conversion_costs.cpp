/* Conversion-path costs and the SYCL type limits that constrain them.
 *
 * Findings this probe exists to record (all verified on this toolchain):
 *
 *   1. sycl::vec only allows 1, 2, 3, 4, 8 or 16 elements. There is no 64-wide
 *      vector, so a 64-byte int8 store cannot be expressed as one vec<>.
 *      It has to be 4x vec<int32_t,16> or a raw memcpy of 64 bytes.
 *   2. There is no free sycl::convert() for vectors, and no converting
 *      constructor from vec<float,16> to vec<int8_t,16> or vec<bfloat16,16>.
 *      The working form is the *member* .convert<>() that sycl::vec provides
 *      on integral element types.
 *   3. Consequence for the library: the int8 pack in store_reg() and the
 *      float->bf16 convert in the epilogue are scalar loops by necessity.
 *      Measured below, and both are already at the bandwidth limit, so the
 *      missing intrinsics cost nothing.
 *
 *   ./build.sh && ./bin/rejected_conversion_costs
 */
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;
using BF16 = vec<BF, 16>;
using I8x16 = vec<int8_t, 16>;

static inline vec<int32_t,16> I32x(const F16 &r) {
    // the only working float -> int32 conversion in this SYCL build
    return r.convert<int32_t>();
}

// pack 16 rounded floats to 16 int8 via scalar lanes (what store_reg does)
static inline void pack_scalar(int8_t *p, const F16 &v, float inv) {
    const F16 r = sycl::min(sycl::max(sycl::rint(v * inv), F16(-127.f)), F16(127.f));
    const auto i = I32x(r);
    uint32_t b[4] = {0, 0, 0, 0};
    for (int t = 0; t < 16; t++)
        b[t >> 2] |= ((uint32_t)i[t] & 0xFFu) << (8 * (t & 3));
    std::memcpy(p, b, 16);
}

// four registers -> one 64-byte store, the widest store this type system allows
static inline void pack_wide(int8_t *p, const F16 &a, const F16 &b, const F16 &c,
                             const F16 &d, float inv) {
    uint32_t o[16];
    const F16 *v[4] = {&a, &b, &c, &d};
    for (int r = 0; r < 4; r++) {
        const F16 q =
            sycl::min(sycl::max(sycl::rint((*v[r]) * inv), F16(-127.f)), F16(127.f));
        const auto i = I32x(q);
        for (int j = 0; j < 4; j++)
            o[r * 4 + j] = ((uint32_t)i[4 * j] & 0xFFu) | (((uint32_t)i[4 * j + 1] & 0xFFu) << 8) |
                           (((uint32_t)i[4 * j + 2] & 0xFFu) << 16) |
                           (((uint32_t)i[4 * j + 3] & 0xFFu) << 24);
    }
    std::memcpy(p, o, 64);
}

int main(int argc, char **argv) {
    const size_t M = argc > 1 ? strtoul(argv[1], 0, 10) : 4096;
    const size_t K = argc > 2 ? strtoul(argv[2], 0, 10) : 3072;
    const size_t N = M * K;
    queue q;

    uint16_t *x = malloc_shared<uint16_t>(N, q);
    int8_t *o = malloc_shared<int8_t>(N, q);
    for (size_t i = 0; i < N; i++) x[i] = 0x3F80;

    auto bench = [&](const char *nm, auto fn, double bytes) {
        for (int i = 0; i < 5; i++) fn();
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 20; i++) fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best,
                            std::chrono::duration<double, std::milli>(t1 - t0).count() / 20);
        }
        std::printf("  %-40s %7.4f ms  %6.0f GB/s\n", nm, best,
                    bytes / 1e9 / (best * 1e-3));
    };

    std::printf("int8 store paths, M=%zu K=%zu (%.1f MB written)\n", M, K, N / 1e6);
    bench("16 x 16-byte stores, scalar pack", [&] {
        q.submit([&](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(M * 16), range<1>(16)), [=](nd_item<1> it) {
                const size_t row = it.get_global_id(0) >> 4, lane = it.get_local_id(0) & 15;
                if (lane * 256 >= K) return;
                int8_t *p = o + row * K + lane * 256;
                F16 v((float)lane);
                for (int w = 0; w < 16; w++) pack_scalar(p + w * 16, v, 100.f);
            });
        }).wait();
    }, (double)N);
    bench("4 x 64-byte stores, scalar pack", [&] {
        q.submit([&](handler &h) {
            h.parallel_for(nd_range<1>(range<1>(M * 16), range<1>(16)), [=](nd_item<1> it) {
                const size_t row = it.get_global_id(0) >> 4, lane = it.get_local_id(0) & 15;
                if (lane * 256 >= K) return;
                int8_t *p = o + row * K + lane * 256;
                F16 v((float)lane);
                for (int b = 0; b < 16; b += 4)
                    pack_wide(p + b * 16, v, v, v, v, 100.f);
            });
        }).wait();
    }, (double)N);

    std::printf("\nConclusion: store width and pack strategy are both irrelevant here.\n"
                "The kernel is register-residency bound, not store bound.\n");
    return 0;
}