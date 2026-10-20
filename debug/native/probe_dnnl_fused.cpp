/*
 * Can oneDNN fuse our dequant epilogue into the int8 matmul?
 *
 * Our epilogue is
 *     out[m,n] = bias[n] + x_scale[m] * w_scale[n] * sum_k q[m,k]*w[n,k]
 * which today costs a separate kernel and a full int32 round trip through
 * DRAM (read M*N*4, write M*N*2) -- 239 ms/step in the MossImage model.
 *
 * oneDNN's generic scale attributes can express exactly that: a per-row
 * (per-OD) scale on SRC and a per-output-channel (per-OD) scale on WEIGHTS
 * multiply into the accumulator, so the dst scale stays 1 and the bias is
 * added on top.  This probe checks
 *   1. whether the GPU (SYCL) backend accepts that scale combination at all,
 *   2. whether the result matches the reference,
 *   3. how fast the fused primitive is.
 *
 * It links against the oneDNN 3.9 (2025.3) runtime extracted from Intel's
 * apt repo, which links the same libsycl.so.8 the venv's torch uses, so a
 * queue can be handed over directly.
 *
 *   icpx -O3 -fsycl -std=c++17 probe_dnnl_fused.cpp -o bin/probe_dnnl_fused \
 *       -I<dnnl include> -L<dnnl lib> -ldnnl -Wl,-rpath,<dnnl lib>
 */
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

using namespace sycl;

static double bench(const std::function<void()> &fn, int iters = 30) {
    for (int i = 0; i < 5; i++) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main(int argc, char **argv) {
    const int64_t M = argc > 1 ? atoi(argv[1]) : 3952;
    const int64_t K = argc > 2 ? atoi(argv[2]) : 3072;
    const int64_t N = argc > 3 ? atoi(argv[3]) : 3072;

    setvbuf(stdout, nullptr, _IONBF, 0);
    queue q{sycl::gpu_selector_v};
    std::printf("device : %s\n", q.get_device().get_info<info::device::name>().c_str());
    std::printf("M=%lld K=%lld N=%lld\n", (long long)M, (long long)K, (long long)N);

    auto *src = malloc_device<int8_t>(M * K, q);
    auto *wgt = malloc_device<int8_t>(N * K, q);   // checkpoint layout [N,K]
    auto *xs = malloc_device<float>(M, q);
    auto *ws = malloc_device<float>(N, q);
    auto *dst = malloc_device<float>(M * N, q);

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> di(-127, 127);
    std::vector<int8_t> hs(M * K), hw(N * K);
    for (auto &v : hs) v = (int8_t)di(rng);
    for (auto &v : hw) v = (int8_t)di(rng);
    std::vector<float> hxs(M), hws(N);
    for (auto &v : hxs) v = 0.001f + (float)(rng() % 1000) / 1e6f;
    for (auto &v : hws) v = 0.001f + (float)(rng() % 1000) / 1e6f;
    q.memcpy(src, hs.data(), M * K).wait();
    q.memcpy(wgt, hw.data(), N * K).wait();
    q.memcpy(xs, hxs.data(), M * 4).wait();
    q.memcpy(ws, hws.data(), N * 4).wait();

    try {
        auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        auto strm = dnnl::sycl_interop::make_stream(eng, q);

        dnnl::memory::desc md({M, K}, dnnl::memory::data_type::s8, {K, 1});
        dnnl::memory::desc wmd({K, N}, dnnl::memory::data_type::s8, {1, K});  // [K,N] view of [N,K]
        dnnl::memory::desc dmd({M, N}, dnnl::memory::data_type::f32, {N, 1});

        dnnl::primitive_attr attr;
        // mode: which scale configuration to try
        //   0 none | 1 common+common | 2 src per-OD | 3 weights per-OD
        //   4 both per-OD (our epilogue) | 5 src common + weights per-OD
        const int mode = argc > 4 ? atoi(argv[4]) : 4;
        if (mode == 1 || mode == 5)
            attr.set_scales(DNNL_ARG_SRC, 0x0, {1});   // common
        if (mode == 2 || mode == 4)
            attr.set_scales(DNNL_ARG_SRC, 0x1, {M});   // per-row
        if (mode == 3 || mode == 4 || mode == 5)
            attr.set_scales(DNNL_ARG_WEIGHTS, 0x2, {N});  // per-output-channel
        if (mode == 1)
            attr.set_scales(DNNL_ARG_WEIGHTS, 0x0, {1});
        std::printf("mode %d\n", mode);

        dnnl::matmul::primitive_desc pd(eng, md, wmd, dmd, attr);
        dnnl::primitive prim(pd);
        std::printf("primitive created OK\n");

        auto mem_src = dnnl::memory(md, eng, (void *)src);
        auto mem_wgt = dnnl::memory(wmd, eng, (void *)wgt);
        auto mem_dst = dnnl::memory(dmd, eng, (void *)dst);
        auto *xs1 = malloc_device<float>(1, q);
        q.memcpy(xs1, hxs.data(), 4).wait();
        auto *ws1 = malloc_device<float>(1, q);
        q.memcpy(ws1, hws.data(), 4).wait();
        auto mem_xs = (mode == 1 || mode == 5)
            ? dnnl::memory(dnnl::memory::desc({1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), eng, (void *)xs1)
            : dnnl::memory(dnnl::memory::desc({M}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), eng, (void *)xs);
        auto mem_ws = (mode == 1)
            ? dnnl::memory(dnnl::memory::desc({1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), eng, (void *)ws1)
            : dnnl::memory(dnnl::memory::desc({N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x), eng, (void *)ws);

        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, mem_src},
            {DNNL_ARG_WEIGHTS, mem_wgt},
            {DNNL_ARG_DST, mem_dst},
        };
        if (mode == 1 || mode == 2 || mode == 4 || mode == 5)
            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = mem_xs;
        if (mode == 1 || mode == 3 || mode == 4 || mode == 5)
            args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = mem_ws;

        prim.execute(strm, args);
        strm.wait();

        std::vector<float> hd(M * N);
        q.memcpy(hd.data(), dst, M * N * 4).wait();

        double worst = 0;
        int checked = 0;
        for (int idx = 0; idx < 40; idx++) {
            int64_t m = (int64_t)idx * 977 % M, n = (int64_t)idx * 613 % N;
            int32_t a = 0;
            for (int64_t k = 0; k < K; k++) a += (int32_t)hs[m * K + k] * (int32_t)hw[n * K + k];
            double want = (double)a * hxs[m] * hws[n];
            double got = hd[m * N + n];
            double rel = std::fabs(want - got) / (std::fabs(want) + 1e-30);
            if (rel > worst) worst = rel;
            checked++;
        }
        std::printf("checked %d elements, worst relative error vs reference = %.3e\n",
                    checked, worst);

        double ms = bench([&] { prim.execute(strm, args); strm.wait(); });
        double flop = 2.0 * (double)M * (double)N * (double)K;
        std::printf("fused oneDNN matmul: %.4f ms   %.1f TOPS\n", ms, flop / ms * 1e-9);
    } catch (const dnnl::error &e) {
        std::printf("oneDNN error: %s (status %d)\n", e.what(), (int)e.status);
        return 1;
    } catch (const std::exception &e) {
        std::printf("exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
