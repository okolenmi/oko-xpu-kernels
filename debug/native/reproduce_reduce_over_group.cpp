// Minimal reproducer: sycl::reduce_over_group() over a work-group segfaults the
// *host* process on this runtime (Arc B580, Linux, oneAPI 2025.3/2026.0).
//
// Origin: debug/native/probe_ffn_situ.cpp had the FFN row-max computed with
//   sycl::reduce_over_group(it.get_group(), vm, sycl::maximum<>())
// instead of the local-memory + butterfly form.  Timing that variant killed the
// process (SIGSEGV, no SYCL exception, twice) before it printed a number; the
// variant was dropped from the probe.  This file isolates the call so the crash
// can be reported upstream and re-tested after a driver/compiler update.
//
// Build (same toolchain as the other native probes):
//   ccache icpx -O3 -fsycl -std=c++17 reproduce_reduce_over_group.cpp -o bin/reproduce_reduce_over_group
//   LD_LIBRARY_PATH=... ./bin/reproduce_reduce_over_group
//
// Exit status: 0 = every case completed and matched the CPU result,
//              139 (128+SIGSEGV) = the crash reproduced,
//              1 = a result was wrong (not a crash, but also a bug).
//
// It prints one line per case *before* the case runs and flushes, so the last
// line on screen names the case that died.

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char *kNames[] = {"float maximum", "int32 maximum", "float sum",
                               "uint32 min"};
// Rounded up per case: nd_range needs the global range to be a multiple of the
// work-group size, and 192 is not a power of two.
static size_t grid_for(size_t wg) { return ((1u << 16) + wg - 1) / wg * wg; }

// One reduction per case, written back by lane 0 of every work-group so the
// result is observable (and so the compiler cannot delete the reduction).
template <typename T, int OP>
static void reduce_kernel(sycl::queue &q, const std::vector<T> &in,
                          std::vector<T> &out, size_t wg, size_t grid) {
    sycl::buffer<T, 1> bi{const_cast<T *>(in.data()), in.size()};
    sycl::buffer<T, 1> bo{out.data(), out.size()};
    q.submit([&](sycl::handler &h) {
        sycl::accessor a{bi, h, sycl::read_only};
        sycl::accessor b{bo, h, sycl::write_only};
        h.parallel_for(sycl::nd_range<1>(grid, wg), [=](sycl::nd_item<1> it) {
            const T v = a[it.get_global_id(0)];
            T r;
            if constexpr (OP == 0) {
                r = sycl::reduce_over_group(it.get_group(), v, sycl::maximum<>());
            } else if constexpr (OP == 1) {
                r = sycl::reduce_over_group(it.get_group(), v, sycl::minimum<>());
            } else if constexpr (OP == 2) {
                r = sycl::reduce_over_group(it.get_group(), v, sycl::plus<>());
            } else {
                r = sycl::reduce_over_group(it.get_group(), v,
                                            sycl::bit_or<>());
            }
            if (it.get_local_id(0) == 0) b[it.get_global_id(0)] = r;
        });
    });
    q.wait_and_throw();
}

template <typename T, int OP>
static int check(const char *label, sycl::queue &q, size_t wg, int reps) {
    const size_t grid = grid_for(wg);
    std::vector<T> in(grid), out(grid, T{});
    // Values chosen so every reduction has a non-trivial, exactly representable
    // answer for a work-group of any size.
    for (size_t i = 0; i < grid; ++i) {
        if constexpr (OP == 2) {
            in[i] = static_cast<T>(1); // sum == wg
        } else if constexpr (OP == 3) {
            in[i] = static_cast<T>(i & 3u); // bit_or -> 3
        } else {
            in[i] = static_cast<T>(i % 7 + 1); // max -> 7, min -> 1
        }
    }

    printf("%-14s group %4zu x %d reps: ", label, wg, reps);
    fflush(stdout);
    for (int r = 0; r < reps; ++r) reduce_kernel<T, OP>(q, in, out, wg, grid);

    for (size_t i = 0; i < grid; i += wg) {
        double got = static_cast<double>(out[i]);
        double want;
        if constexpr (OP == 2)
            want = static_cast<double>(wg);
        else if constexpr (OP == 3)
            want = 3.0;
        else if constexpr (OP == 0)
            want = 7.0;
        else
            want = 1.0;
        if (got != want) {
            printf("WRONG at %zu: got %.0f want %.0f\n", i, got, want);
            fflush(stdout);
            return 1;
        }
    }
    printf("ok\n");
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    int reps = 50;
    for (int i = 1; i < argc; ++i) {
        if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    }

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const auto dev = q.get_device();
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("driver: %s\n",
           dev.get_info<sycl::info::device::driver_version>().c_str());
    printf("max_work_group_size: %zu\n",
           dev.get_info<sycl::info::device::max_work_group_size>());
    printf("sub_group_sizes:");
    for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>())
        printf(" %zu", s);
    printf("\n");
    fflush(stdout);

    // 192 is the width the FFN kernel used (K/GS sub-groups x SUBG, 3072/256*16).
    const size_t sizes[] = {16, 32, 64, 128, 192, 256, 512, 1024};
    int rc = 0;
    for (size_t wg : sizes) {
        if (wg > dev.get_info<sycl::info::device::max_work_group_size>()) {
            printf("group %4zu: skipped (over device limit)\n", wg);
            continue;
        }
        rc |= check<float, 0>("float maximum", q, wg, reps);
        rc |= check<int32_t, 1>("int32 maximum", q, wg, reps);
        rc |= check<float, 2>("float sum", q, wg, reps);
        rc |= check<uint32_t, 3>("uint32 bit_or", q, wg, reps);
    }
    printf("%s\n", rc ? "FAILED (wrong results)" : "all cases ok");
    return rc;
}
