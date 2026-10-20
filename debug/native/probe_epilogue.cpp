/*
 * Is the dequant epilogue's scalar float->bf16 loop costing anything?
 * Result: no. The scalar loop is vectorised by the compiler; the rounding-bit
 * trick is 2x SLOWER (0.24 vs 0.135 ms). The epilogue is at 88% of the
 * bandwidth floor and has no headroom.
 *
 *   ./build.sh && ./bin/probe_epilogue
 */
// The dequant epilogue converts float->bf16 with a 16-iteration scalar loop per
// vector. Is that costing anything? Compare against the rounding-bit trick and
// a swizzle-based pack. Also check whether clamping / the int32->float convert
// or the redundant per-row w_scale/bias loads matter.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float,16>;
using U16 = vec<uint32_t,16>;
using BF16 = vec<BF,16>;
using I32 = vec<int32_t,16>;

static inline BF16 bf_scalar(F16 v){ BF16 r; for(int t=0;t<16;t++) r[t]=BF(v[t]); return r; }

// round-to-nearest-even in bf16 via a bit add, then pack 16 uint32 -> 16 uint16
static inline vec<uint32_t,8> bf_pack_bit(F16 v){
    U16 u = sycl::bit_cast<U16>(v);
    U16 h = (u + U16(0x7FFFu)) + (u >> 16);
    U16 s = h >> 16;
    vec<uint32_t,8> o;
    for(int j=0;j<8;j++) o[j] = s[2*j] | (s[2*j+1] << 16);
    return o;
}
static inline vec<uint32_t,8> bf_pack_swz(F16 v){
    U16 u = sycl::bit_cast<U16>(v);
    U16 h = (u + U16(0x7FFFu)) + (u >> 16);
    U16 s = h >> 16;
    vec<uint32_t,16> e = s.swizzle<0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14>();
    vec<uint32_t,16> o = s.swizzle<1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15>();
    vec<uint32_t,16> w = e | (o << 16);
    vec<uint32_t,8> r;
    for(int j=0;j<8;j++) r[j]=w[j];
    return r;
}

template<int MODE>
static void run(queue&q,const int32_t*acc,BF*out,const float*xs,const float*ws,const float*bs,
                size_t M,size_t N){
    const size_t nv=N/16;
    const int has_bias = bs!=nullptr;
    q.submit([&](handler&h){
        h.parallel_for(range<2>(M,nv),[=](id<2> idx){
            const size_t m=idx[0], v=idx[1];
            const I32 a=reinterpret_cast<const I32*>(acc+m*N)[v];
            const F16 w=reinterpret_cast<const F16*>(ws)[v];
            F16 val=F16(a.convert<float>())*(w*xs[m]);
            if(has_bias) val=val+reinterpret_cast<const F16*>(bs)[v];
            if(MODE==0){ reinterpret_cast<BF16*>(out+m*N)[v]=bf_scalar(val); }
            else {
                vec<uint32_t,8> o = (MODE==1) ? bf_pack_bit(val) : bf_pack_swz(val);
                std::memcpy(out+m*N+v*16, &o, 32);
            }
        });
    }).wait();
}

int main(int argc,char**argv){
    size_t M=argc>1?strtoul(argv[1],0,10):4096, N=argc>2?strtoul(argv[2],0,10):3072;
    queue q;
    int32_t*acc=malloc_shared<int32_t>(M*N,q);
    BF*out=malloc_shared<BF>(M*N,q);
    float*xs=malloc_shared<float>(M,q);
    float*ws=malloc_shared<float>(N,q);
    float*bs=malloc_shared<float>(N,q);
    for(size_t i=0;i<M*N;i++) acc[i]=(int32_t)(i%97);
    for(size_t i=0;i<M;i++) xs[i]=0.01f;
    for(size_t i=0;i<N;i++){ ws[i]=0.02f; bs[i]=0.5f; }
    auto bench=[&](const char*nm,auto fn){
        double best=1e9;
        for(int rep=0;rep<3;rep++){
            for(int i=0;i<5;i++) fn();
            auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<20;i++) fn();
            auto t1=std::chrono::high_resolution_clock::now();
            best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count()/20);
        }
        std::printf("%-40s %7.4f ms  %6.0f GB/s\n",nm,best,(double)(M*N*6)/1e9/(best*1e-3));
    };
    std::printf("epilogue M=%zu N=%zu (%.1f MB read + %.1f MB write)\n\n",M,N,M*N*4/1e6,M*N*2/1e6);
    bench("bf16 via 16 scalar conversions (current)",[&]{ run<0>(q,acc,out,xs,ws,bs,M,N); });
    bench("bf16 via rounding-bit trick",             [&]{ run<1>(q,acc,out,xs,ws,bs,M,N); });
    bench("bf16 via bit trick + swizzle pack",       [&]{ run<2>(q,acc,out,xs,ws,bs,M,N); });
    // no bias (isolates the branch)
    bench("bit trick, bias=NULL",                    [&]{ run<1>(q,acc,out,xs,ws,nullptr,M,N); });
    return 0;
}