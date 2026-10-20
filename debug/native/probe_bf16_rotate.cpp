/*
 * The BF16 (rotate-only) kernel. Establishes that doing all four axis-3
 * butterflies first and then one flat 16-register store loop is 1.44x faster
 * (0.315 vs 0.454 ms) than interleaving butterflies with stores -- and that
 * the interleaved version also produced wrong results.
 *
 *   ./build.sh && ./bin/probe_bf16_rotate 4096 3072
 */
// The BF16 rotate (no row-scale dependency) came out at 0.92 ms, SLOWER than
// rotate+quantize at 0.51 ms -- even though it should have the better register
// behaviour. It moves 50.4 MB vs 37.8 MB (1.33x) but takes 1.79x as long.
// Which store pattern is responsible?
//
//   A: axis-3 bfly, then convert+store each 4-register group immediately
//      (what is in the library now)
//   B: all four axis-3 bfly first, then one flat 16-register convert+store loop
//      (same shape as the int8 kernel)
//   C: like B but the bf16 conversion is hoisted into a 16-wide pass
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
using namespace sycl;
using BF = sycl::ext::oneapi::bfloat16;
using F16 = vec<float,16>;
using U16 = vec<uint32_t,16>;
using BF16 = vec<BF,16>;

constexpr int GS=256, NREG=16, SUBG=16;
static const float H4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
static inline F16 h4_col(int c,bool d1){F16 v;for(int t=0;t<16;t++)v[t]=0.5f*H4[d1?((t>>2)&3):(t&3)][c];return v;}
static inline F16 ld(const uint16_t*p){
    const vec<uint32_t,8> u=reinterpret_cast<const vec<uint32_t,8>*>(p)[0];
    U16 w;for(int j=0;j<8;j++){w[2*j]=u[j]<<16;w[2*j+1]=u[j]&0xFFFF0000u;}
    return sycl::bit_cast<F16>(w);
}
static inline void bfly4(F16&a0,F16&a1,F16&a2,F16&a3){
    const F16 p=a0+a1,m=a0-a1,q=a2+a3,n=a2-a3;
    a0=(p+n)*0.5f;a1=(p-n)*0.5f;a2=(m+q)*0.5f;a3=(q-m)*0.5f;
}
static inline F16 inreg(F16 v,const F16&c0,const F16&c1,const F16&c2,const F16&c3,
                        const F16&d0,const F16&d1,const F16&d2,const F16&d3){
    F16 t=c0*v.swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>();
    t=t+c1*v.swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>();
    t=t+c2*v.swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>();
    t=t+c3*v.swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
    F16 u=d0*t.swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>();
    u=u+d1*t.swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>();
    u=u+d2*t.swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>();
    u=u+d3*t.swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
    return u;
}
static inline void store_bf(BF* p, F16 v){
    BF16 r; for(int t=0;t<16;t++) r[t]=BF(v[t]);
    reinterpret_cast<BF16*>(p)[0]=r;
}

template<int MODE>
static void run(queue&q,const uint16_t*x,BF*xr,size_t M,size_t K){
    const size_t ng=K/GS;
    const F16 c0=h4_col(0,0),c1=h4_col(1,0),c2=h4_col(2,0),c3=h4_col(3,0);
    const F16 d0=h4_col(0,1),d1=h4_col(1,1),d2=h4_col(2,1),d3=h4_col(3,1);
    q.submit([&](handler&h){
        h.parallel_for(nd_range<1>(range<1>(M*SUBG),range<1>(SUBG)),[=](nd_item<1>it){
            const size_t row=it.get_global_id(0)>>4, lane=it.get_local_id(0)&15;
            if(lane>=ng) return;
            const uint16_t*p=x+row*K+lane*GS;
            BF*d=xr+row*K+lane*GS;
            F16 W[NREG];
            for(int b=0;b<NREG;b+=4){
                for(int j=0;j<4;j++) W[b+j]=ld(p+(b+j)*16);
                for(int j=0;j<4;j++) W[b+j]=inreg(W[b+j],c0,c1,c2,c3,d0,d1,d2,d3);
                bfly4(W[b],W[b+1],W[b+2],W[b+3]);
            }
            if(MODE==0){                       // A: interleave bfly and store
                for(int c=0;c<4;c++){
                    bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                    for(int u=0;u<4;u++) store_bf(d+c+4*u, W[c+4*u]);
                }
            } else {                           // B/C: all bfly, then flat store
                for(int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                for(int w=0;w<NREG;w++) store_bf(d+w, W[w]);
            }
        });
    }).wait();
}

int main(int argc,char**argv){
    size_t M=argc>1?strtoul(argv[1],0,10):4096, K=argc>2?strtoul(argv[2],0,10):3072;
    queue q;
    uint16_t*x=malloc_shared<uint16_t>(M*K,q);
    BF*xr=malloc_shared<BF>(M*K,q);
    for(size_t i=0;i<M*K;i++) x[i]=0x3F80;
    auto bench=[&](const char*nm,auto fn){
        double best=1e9;
        for(int rep=0;rep<3;rep++){
            for(int i=0;i<5;i++) fn();
            auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<20;i++) fn();
            auto t1=std::chrono::high_resolution_clock::now();
            best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count()/20);
        }
        std::printf("%-44s %7.4f ms  %6.0f GB/s\n",nm,best,(double)M*K*4/1e9/(best*1e-3));
    };
    std::printf("M=%zu K=%zu  bf16 rotate: 25.2 MB read + 25.2 MB write\n",M,K);
    bench("A: bfly/store interleaved (library)",[&]{ run<0>(q,x,xr,M,K); });
    bench("B: all bfly, then flat store",        [&]{ run<1>(q,x,xr,M,K); });
    return 0;
}