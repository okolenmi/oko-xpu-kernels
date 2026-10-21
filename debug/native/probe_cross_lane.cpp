// What does a cross-lane exchange cost, and can the rotate's last open shape
// actually be built?
//
// docs/performance.md says the rotate is 3.1x off its floor because it has to
// hold 256 floats (16 zmm) live across load -> row abs-max -> quantize, and
// that "closing it needs one group split across 4 lanes with a single
// cross-lane axis".  This probe measures the two numbers that decide it:
//
//   1. the cost of one cross-lane exchange -- scalar and vector -- so the
//      "16 select_from_group calls per 512-bit exchange" can be turned into
//      nanoseconds;
//   2. how many exchanges the split would need per row, at the shapes the
//      model uses.
//
// The structural part is in the comment at the bottom: at K=3072 the row has
// 12 ConvRot groups and the sub-group has 16 lanes, so a group split across
// 4 lanes needs 48 lane-slots for one row -- three sub-groups -- and the
// row's abs-max would have to cross them.
//
//   ./build.sh && ./bin/probe_cross_lane
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace sycl;
using F16 = vec<float, 16>;

static double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main() {
    queue q{sycl::gpu_selector_v};
    const size_t WG = 64, GROUPS = 1024;
    const int R = 4096;

    // -- 1. the cost of one exchange ---------------------------------------
    // scalar: the pattern the kernel already uses for the row's abs-max.
    {
        buffer<float, 1> out{GROUPS * WG};
        q.submit([&](handler &h) {
            sycl::accessor a{out, h, sycl::write_only};
            h.parallel_for(nd_range<1>(GROUPS * WG, WG), [=](nd_item<1> it) {
                auto sg = it.get_sub_group();
                const uint32_t lane = it.get_local_id(0) & 15;
                float v = static_cast<float>(lane);
                for (int i = 0; i < R; i++) {
                    // a data-dependent lane id, so the compiler cannot fold it
                    v = sycl::select_from_group(sg, v, (lane + i + 1) & 15);
                }
                if (lane == 0) a[it.get_global_id(0)] = v;
            });
        });
        q.wait_and_throw();
        double t0 = now_s();
        q.submit([&](handler &h) {
            sycl::accessor a{out, h, sycl::write_only};
            h.parallel_for(nd_range<1>(GROUPS * WG, WG), [=](nd_item<1> it) {
                auto sg = it.get_sub_group();
                const uint32_t lane = it.get_local_id(0) & 15;
                float v = static_cast<float>(lane);
                for (int i = 0; i < R; i++) {
                    v = sycl::select_from_group(sg, v, (lane + i + 1) & 15);
                }
                if (lane == 0) a[it.get_global_id(0)] = v;
            });
        });
        q.wait_and_throw();
        double dt = now_s() - t0;
        // WG lanes work in parallel; each lane does R exchanges
        double per_exchange_ns = dt * 1e9 / (R * WG);
        std::printf("scalar select_from_group : %8.2f ns per call (per lane)\n",
                    per_exchange_ns);
        std::printf("  -> a 16-element exchange, if it lowers to 16 calls: %8.2f ns\n",
                    per_exchange_ns * 16);
    }

    // -- 2. the vector form ------------------------------------------------
    {
        buffer<float, 1> out{GROUPS * WG * 16};
        q.submit([&](handler &h) {
            sycl::accessor a{out, h, sycl::write_only};
            h.parallel_for(nd_range<1>(GROUPS * WG, WG), [=](nd_item<1> it) {
                auto sg = it.get_sub_group();
                const uint32_t lane = it.get_local_id(0) & 15;
                F16 v;
                for (int t = 0; t < 16; t++) v[t] = static_cast<float>(lane + t);
                for (int i = 0; i < R; i++) {
                    v = sycl::select_from_group(sg, v, (lane + i + 1) & 15);
                }
                if (lane == 0)
                    for (int t = 0; t < 16; t++)
                        a[it.get_global_id(0) * 16 + t] = v[t];
            });
        });
        q.wait_and_throw();
        double t0 = now_s();
        q.submit([&](handler &h) {
            sycl::accessor a{out, h, sycl::write_only};
            h.parallel_for(nd_range<1>(GROUPS * WG, WG), [=](nd_item<1> it) {
                auto sg = it.get_sub_group();
                const uint32_t lane = it.get_local_id(0) & 15;
                F16 v;
                for (int t = 0; t < 16; t++) v[t] = static_cast<float>(lane + t);
                for (int i = 0; i < R; i++) {
                    v = sycl::select_from_group(sg, v, (lane + i + 1) & 15);
                }
                if (lane == 0)
                    for (int t = 0; t < 16; t++)
                        a[it.get_global_id(0) * 16 + t] = v[t];
            });
        });
        q.wait_and_throw();
        double dt = now_s() - t0;
        double per_exchange_ns = dt * 1e9 / (R * WG);
        std::printf("vector select_from_group : %8.2f ns per 512-bit exchange\n",
                    per_exchange_ns);
    }

    // -- 3. what the split would need --------------------------------------
    std::printf("\nthe split, per row (one group = 256 elements, 4 lanes each):\n");
    for (int K : {1024, 3072, 8192}) {
        const int groups = K / 256;             // groups per row
        const int slots = groups * 4;            // lane-slots the row needs
        const int sub_groups = (slots + 15) / 16; // sub-groups of 16 lanes
        const int quarter_groups_per_lane = (groups * 4 + 15) / 16;  // per lane
        // each lane holds 4 registers per quarter-group and must obtain 3
        // other lanes' registers: 12 vector exchanges per group it touches
        const long exchanges_per_lane = (long)quarter_groups_per_lane * 12;
        std::printf(
            "  K=%5d: %2d groups/row -> %2d lane-slots -> %d sub-group(s) %s"
            " ; %2d quarter-groups/lane -> %3ld exchanges/lane/row\n",
            K, groups, slots, sub_groups,
            sub_groups > 1 ? "(the row's abs-max would have to CROSS them)" : "",
            quarter_groups_per_lane, exchanges_per_lane);
    }
    std::printf(
        "\nfor reference: the rotate at M=4096, K=3072 is 0.52 ms per call, and\n"
        "its structural floor is 0.27 ms, so the prize is ~0.25 ms per row.\n");
    return 0;
}
