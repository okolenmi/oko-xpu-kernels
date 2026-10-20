/*
 * Attribution ablation for the rotate+quantize kernel.
 * Each MODE removes exactly one piece of the production kernel so the cost of
 * each can be read off directly. The full kernel writes its output ONCE -- an
 * earlier version of this probe stored twice and inflated the "store" number
 * to 0.215 ms, which sent the optimisation effort after the wrong thing.
 *
 *   ./build.sh && ./bin/probe_rotate_ablation 4096 3072
 */
// Corrected ablation: ONE store pass (the previous version stored twice).
// Also reports cache sizes, because a 37.6 MB working set may be L2-resident
// and that changes what the "memory floor" actually is.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
using namespace sycl;
using F16 = vec<float,16>;
using U16 = vec<uint32_t,16>;
using I32 = vec<int32_t,16>;

constexpr int GS=256, NREG=16, SUBG=16;
static const float H4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
static inline F16 h4_col(int c,bool d1){F16 v;for(int t=0;t<16;t++)v[t]=0.5f*H4[d1?((t>>2)&3):(t&3)][c];return v;}
static inline F16 ld(const uint16_t*p){
    const vec<uint32_t,8> u=reinterpret_cast<const vec<uint32_t,8>*>(p)[0];
    U16 w;for(int j=0;j<8;j++){w[2*j]=u[j]<<16;w[2*j+1]=u[j]&0xFFFF0000u;}
    return sycl::bit_cast<F16>(w);
}
static inline void pk_store(int8_t*p,F16 v,float inv){
    const F16 r=sycl::min(sycl::max(sycl::rint(v*inv),F16(-127.f)),F16(127.f));
    uint32_t b[4]={0,0,0,0};
    for(int t=0;t<16;t++) b[t>>2]|=(uint32_t)(uint8_t)(int8_t)(int32_t)r[t]<<(8*(t&3));
    std::memcpy(p,b,16);
}
static inline void pk_store_i32(int8_t*p,F16 v,float inv){
    const I32 i=sycl::min(sycl::max(sycl::rint(v*inv),F16(-127.f)),F16(127.f)).convert<int32_t>();
    uint32_t b[4];
    for(int j=0;j<4;j++)
        b[j]=((uint32_t)i[4*j]&0xFFu)|(((uint32_t)i[4*j+1]&0xFFu)<<8)
            |(((uint32_t)i[4*j+2]&0xFFu)<<16)|(((uint32_t)i[4*j+3]&0xFFu)<<24);
    std::memcpy(p,b,16);
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

// MODE 0 full | 1 no store | 2 no transform | 3 no absmax | 4 int32 pack | 5 no sub-group reduce
template<int MODE>
static void run(queue&q,const uint16_t*x,int8_t*o,float*sc,size_t M,size_t K){
    const size_t ng=K/GS;
    const F16 c0=h4_col(0,0),c1=h4_col(1,0),c2=h4_col(2,0),c3=h4_col(3,0);
    const F16 d0=h4_col(0,1),d1=h4_col(1,1),d2=h4_col(2,1),d3=h4_col(3,1);
    q.submit([&](handler&h){
        h.parallel_for(nd_range<1>(range<1>(M*SUBG),range<1>(SUBG)),[=](nd_item<1>it){
            const size_t row=it.get_global_id(0)>>4, lane=it.get_local_id(0)&15;
            const uint16_t*xr=x+row*K;
            const size_t grp=lane;
            const bool act=(grp<ng);
            F16 W[NREG]; F16 amaxv(0.f);
            if(act){
                const uint16_t*p=xr+grp*GS;
                for(int b=0;b<NREG;b+=4){
                    for(int j=0;j<4;j++) W[b+j]=ld(p+(b+j)*16);
                    if(MODE!=2){
                        for(int j=0;j<4;j++) W[b+j]=inreg(W[b+j],c0,c1,c2,c3,d0,d1,d2,d3);
                        bfly4(W[b],W[b+1],W[b+2],W[b+3]);
                    }
                }
                if(MODE!=2) for(int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                if(MODE!=3) for(int w=0;w<NREG;w++) amaxv=sycl::max(amaxv,sycl::fabs(W[w]));
            }
            if(MODE==5){
                // reduce but do NOT write scale / do not use it
                F16 r0=sycl::max(amaxv,amaxv.swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());
                F16 r1=sycl::max(r0,r0.swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>());
                F16 r2=sycl::max(r1,r1.swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>());
                F16 r3=sycl::max(r2,r2.swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>());
                if(lane==0) sc[row]=r3[0];
                if(act) for(int w=0;w<NREG;w++) pk_store(o+row*K+grp*GS+w*16,W[w],1.f);
                return;
            }
            F16 r0=sycl::max(amaxv,amaxv.swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());
            F16 r1=sycl::max(r0,r0.swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>());
            F16 r2=sycl::max(r1,r1.swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>());
            F16 r3=sycl::max(r2,r2.swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>());
            const auto sg=it.get_sub_group(); float v=r3[0];
            v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(lane^8)));
            v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(lane^4)));
            v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(lane^2)));
            v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(lane^1)));
            const float s=sycl::fmax(v/127.f,1e-30f);
            if(lane==0) sc[row]=s;
            if(MODE==1) return;
            if(!act) return;
            const float inv=1.f/s; int8_t*d=o+row*K+grp*GS;
            if(MODE==4) for(int w=0;w<NREG;w++) pk_store_i32(d+w*16,W[w],inv);
            else        for(int w=0;w<NREG;w++) pk_store(d+w*16,W[w],inv);
        });
    }).wait();
}

int main(int argc,char**argv){
    queue q;
    {
        auto d=q.get_device();
        std::printf("device      : %s\n", d.get_info<info::device::name>().c_str());
        std::printf("L2 cache    : %.1f MB\n", d.get_info<info::device::global_mem_cache_size>()/1048576.0);
        std::printf("local mem   : %.1f KB\n", d.get_info<info::device::local_mem_size>()/1024.0);
        std::printf("max WG size : %zu\n", (size_t)d.get_info<info::device::max_work_group_size>());
        std::printf("samplers    : %u\n", (unsigned)d.get_info<info::device::max_num_sub_groups>());
    }
    size_t M=argc>1?strtoul(argv[1],0,10):4096, K=argc>2?strtoul(argv[2],0,10):3072;
    size_t N=M*K;
    uint16_t*x=malloc_shared<uint16_t>(N,q);
    int8_t*o=malloc_shared<int8_t>(N,q);
    float*sc=malloc_shared<float>(M,q);
    for(size_t i=0;i<N;i++)x[i]=0x3F80;
    auto bench=[&](const char*nm,auto fn){
        double best=1e9;
        for(int rep=0;rep<3;rep++){
            for(int i=0;i<5;i++) fn();
            auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<20;i++) fn();
            auto t1=std::chrono::high_resolution_clock::now();
            best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count()/20);
        }
        std::printf("%-40s %7.4f ms\n",nm,best); return best;
    };
    std::printf("\nM=%zu K=%zu  (25.2 MB in / 12.6 MB out)\n\n",M,K);
    double a=bench("0 full (production)",          [&]{ run<0>(q,x,o,sc,M,K); });
    double b=bench("1 minus int8 store",          [&]{ run<1>(q,x,o,sc,M,K); });
    double c=bench("2 minus transform",           [&]{ run<2>(q,x,o,sc,M,K); });
    double d=bench("3 minus abs-max/reduce",      [&]{ run<3>(q,x,o,sc,M,K); });
    double e=bench("4 int32-vector pack",         [&]{ run<4>(q,x,o,sc,M,K); });
    double f=bench("5 no sub-group reduce (inv=1)",[&]{ run<5>(q,x,o,sc,M,K); });
    std::printf("\n  transform            = %+.4f ms\n",a-c);
    std::printf("  int8 store           = %+.4f ms\n",a-b);
    std::printf("  abs-max+reduce       = %+.4f ms\n",a-d);
    std::printf("  sub-group reduce     = %+.4f ms\n",a-f);
    std::printf("  int32 pack saves     = %+.4f ms\n",a-e);
    std::printf("  load+store skeleton  = %+.4f ms\n",c);
    return 0;
}