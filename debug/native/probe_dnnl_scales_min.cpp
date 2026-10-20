// Minimal repro: runtime scales on a GPU int8 matmul (oneDNN).
//
//   icpx -O3 -fsycl -std=c++17 probe_dnnl_scales_min.cpp -o bin/probe_dnnl_scales_min \
//       -I<dnnl>/include -L<dnnl>/lib -ldnnl -Wl,-rpath,<dnnl>/lib
//   ./bin/probe_dnnl_scales_min <mode>
//
//   mode 0: no scales
//   mode 1: SRC common            + WEIGHTS common
//   mode 2: SRC per-row (per-M)
//   mode 3:                         WEIGHTS per-output-channel (per-N)
//   mode 4: SRC per-row (per-M)    + WEIGHTS per-output-channel (per-N)   <- the
//           smooth-quantisation / dynamic-per-token epilogue
//   mode 5: SRC common            + WEIGHTS per-output-channel (per-N)
//
// Uses set_scales_mask(arg, mask); the scale values are passed at execution
// time as DNNL_ARG_ATTR_SCALES | arg, which is what the header documents.
//
// Modes 6-8 use the other overload, set_scales(arg, mask, groups), to show
// what the groups argument does to the same requests:
//   mode 6: SRC per-M      as set_scales(SRC, 0x1, {1})
//   mode 7: WEIGHTS per-N  as set_scales(WEIGHTS, 0x2, {1})
//   mode 8: WEIGHTS per-N  as set_scales(WEIGHTS, 0x2, {N})
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <unordered_map>

using namespace sycl;

int main(int argc, char **argv) {
    const int mode = argc > 1 ? atoi(argv[1]) : 4;
    const int64_t M = 512, K = 512, N = 512;

    queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n",
            q.get_device().get_info<info::device::name>().c_str());

    auto *src = malloc_device<int8_t>(M * K, q);
    auto *wgt = malloc_device<int8_t>(N * K, q);
    auto *dst = malloc_device<float>(M * N, q);
    auto *xs = malloc_device<float>(M, q);
    auto *ws = malloc_device<float>(N, q);
    auto *one = malloc_device<float>(1, q);
    q.memset(src, 1, M * K).wait();
    q.memset(wgt, 1, N * K).wait();
    q.memset(xs, 0, M * 4).wait();
    q.memset(ws, 0, N * 4).wait();
    q.memset(one, 0, 4).wait();

    auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, q);

    dnnl::memory::desc md({M, K}, dnnl::memory::data_type::s8, {K, 1});
    dnnl::memory::desc wmd({K, N}, dnnl::memory::data_type::s8, {1, K});  // [K,N] view of [N,K]
    dnnl::memory::desc dmd({M, N}, dnnl::memory::data_type::f32, {N, 1});

    dnnl::primitive_attr attr;
    // For a 2-D SRC [M,K]: mask bit 0 = M, bit 1 = K.
    // For a 2-D WEIGHTS [K,N]: mask bit 0 = K, bit 1 = N.
    if (mode == 1 || mode == 5) attr.set_scales_mask(DNNL_ARG_SRC, 0x0);
    if (mode == 2 || mode == 4) attr.set_scales_mask(DNNL_ARG_SRC, 0x1);
    if (mode == 6) attr.set_scales(DNNL_ARG_SRC, 0x1, {1});
    if (mode == 1 || mode == 3 || mode == 4 || mode == 5)
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0x2);
    if (mode == 7) attr.set_scales(DNNL_ARG_WEIGHTS, 0x2, {1});
    if (mode == 8) attr.set_scales(DNNL_ARG_WEIGHTS, 0x2, {N});
    if (mode == 1) attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0x0);

    std::printf("mode %d\n", mode);
    dnnl::matmul::primitive_desc pd(eng, md, wmd, dmd, attr);
    dnnl::primitive prim(pd);
    std::printf("primitive created OK\n");

    auto val = [&](float *p, int64_t n) {
        return dnnl::memory(
                dnnl::memory::desc({n}, dnnl::memory::data_type::f32,
                        dnnl::memory::format_tag::x),
                eng, p);
    };
    std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, dnnl::memory(md, eng, src)},
            {DNNL_ARG_WEIGHTS, dnnl::memory(wmd, eng, wgt)},
            {DNNL_ARG_DST, dnnl::memory(dmd, eng, dst)},
    };
    if (mode == 1 || mode == 2 || mode == 4 || mode == 5 || mode == 6)
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] =
                (mode == 1 || mode == 5) ? val(one, 1) : val(xs, M);
    if (mode == 1 || mode == 3 || mode == 4 || mode == 5 || mode == 7 || mode == 8)
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
                (mode == 1) ? val(one, 1) : val(ws, N);

    prim.execute(strm, args);
    strm.wait();
    std::printf("executed OK\n");
    return 0;
}
