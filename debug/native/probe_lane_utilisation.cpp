/*
 * At K=3072 only 12 of 16 sub-group lanes have work. Is that 25% wasted?
 * Result: no. Per-element throughput is *highest* at K=1024 (4/16 lanes busy)
 * because the working set drops into L2. Idle lanes are free; the apparent
 * correlation is cache size. Also shows that putting several rows in one
 * work-group is a 1.6-3x loss.
 *
 *   ./build.sh && ./bin/probe_lane_utilisation
 */
// At K=3072 there are 12 ConvRot groups per row but the sub-group is 16 lanes,
// so 4/16 lanes exit immediately. Is that 25% of the kernel wasted?
// Compare per-element throughput at K=3072 (12/16 lanes) against K=4096
// (16/16 lanes), normalising by element count.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
using namespace sycl;
using F16 = vec<float,16>;
using U16 = vec<uint32_t,16>;

constexpr int GS=256, NREG=16, SUBG=16;
static const float H4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
static inline F16 h4_col(int c,bool d1){F16 v;for(int t=0;t<16;t++)v[t]=0.5f*H4[d1?((t>>2)&3):(t&3)][c];return v;}
static inline F16 ld(const uint16_t*p){
    const vec<uint32_t,8> u=reinterpret_cast<const vec<uint32_t,8>*>(p)[0];
    U16 w;for(int j=0;j<8;j++){w[2*j]=u[j]<<16;w[2*j+1]=u[j]&0xFFFF0000u;}
    return sycl::bit_cast<F16>(w);
}
static inline void st(int8_t*p,F16 v,float inv){
    const F16 r=sycl::min(sycl::max(sycl::rint(v*inv),F16(-127.f)),F16(127.f));
    uint32_t b[4]={0,0,0,0};
    for(int t=0;t<16;t++) b[t>>2]|=(uint32_t)(uint8_t)(int8_t)(int32_t)r[t]<<(8*(t&3));
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

// ROWS_PER_WG = how many rows one work-group covers (lanes = ROWS*SUBG).
// With ng groups per row and SUBG lanes, idle lanes = SUBG - ng per row.
template<int RPT>
static void run(queue&q,const uint16_t*x,int8_t*o,float*sc,size_t M,size_t K){
    const size_t ng=K/GS;
    const F16 c0=h4_col(0,0),c1=h4_col(1,0),c2=h4_col(2,0),c3=h4_col(3,0);
    const F16 d0=h4_col(0,1),d1=h4_col(1,1),d2=h4_col(2,1),d3=h4_col(3,1);
    const size_t wg=RPT*SUBG;
    q.submit([&](handler&h){
        h.parallel_for(nd_range<1>(range<1>(M/RPT*wg),range<1>(wg)),[=](nd_item<1>it){
            const size_t rbase=(it.get_global_id(0)/wg)*RPT;
            const size_t lane=it.get_local_id(0)&15;
            for(int r=0;r<RPT;r++){
                const size_t row=rbase+r;
                if(row>=M) return;
                const uint16_t*xr=x+row*K;
                F16 W[NREG]; F16 amaxv(0.f);
                const size_t grp=lane;
                if(grp<ng){
                    const uint16_t*p=xr+grp*GS;
                    for(int b=0;b<NREG;b+=4){
                        for(int j=0;j<4;j++) W[b+j]=ld(p+(b+j)*16);
                        for(int j=0;j<4;j++) W[b+j]=inreg(W[b+j],c0,c1,c2,c3,d0,d1,d2,d3);
                        bfly4(W[b],W[b+1],W[b+2],W[b+3]);
                    }
                    for(int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                    for(int w=0;w<NREG;w++) amaxv=sycl::max(amaxv,sycl::fabs(W[w]));
                }
                F16 a0=sycl::max(amaxv,amaxv.swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());
                F16 a1=sycl::max(a0,a0.swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>());
                F16 a2=sycl::max(a1,a1.swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>());
                F16 a3=sycl::max(a2,a2.swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>());
                auto sg=it.get_sub_group();
                // only reduce within this row's 16 lanes
                float v=a3[0];
                const int sl = (int)(it.get_local_id(0)&15);
                v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(sl^8)));
                v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(sl^4)));
                v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(sl^2)));
                v=sycl::fmax(v,sycl::select_from_group(sg,v,(uint32_t)(sl^1)));
                const float s=sycl::fmax(v/127.f,1e-30f);
                if(sl==0) sc[row]=s;
                if(grp<ng){
                    const float inv=1.f/s; int8_t*d=o+row*K+grp*GS;
                    for(int w=0;w<NREG;w++) st(d+w*16,W[w],inv);
                }
            }
        });
    }).wait();
}

int main(){
    queue q;
    const size_t MEL=4096*4096;   // keep element count constant
    uint16_t*x=malloc_shared<uint16_t>(MEL,q);
    int8_t*o=malloc_shared<int8_t>(MEL,q);
    float*sc=malloc_shared<float>(4096,q);
    for(size_t i=0;i<MEL;i++) x[i]=0x3F80;
    auto bench=[&](size_t M,size_t K,int rpt){
        auto fn=[&]{ switch(rpt){case 1:run<1>(q,x,o,sc,M,K);break;case 2:run<2>(q,x,o,sc,M,K);break;
                                  case 4:run<4>(q,x,o,sc,M,K);break;case 8:run<8>(q,x,o,sc,M,K);break;} };
        double best=1e9;
        for(int rep=0;rep<3;rep++){
            for(int i=0;i<5;i++) fn();
            auto t0=std::chrono::high_resolution_clock::now();
            for(int i=0;i<20;i++) fn();
            auto t1=std::chrono::high_resolution_clock::now();
            best=std::min(best,std::chrono::duration<double,std::milli>(t1-t0).count()/20);
        }
        size_t ng=K/GS;
        std::printf("M=%5zu K=%5zu (%2zu grp/row, %2zu/%2zu lanes busy) rows/wg=%d : "
                    "%7.4f ms   %6.1f Melem/s\n",M,K,ng,ng>16?16:ng,16,rpt,best,(double)M*K/1e6/(best*1e-3));
    };
    std::printf("constant total elements = %.1f Melem\n\n", MEL/1e6);
    bench(4096,4096,1);
    bench(3413,4096,1);
    bench(2048,4096,1);   // K=4096, half the rows
    std::printf("\n--- fewer groups per row -> more idle lanes ---\n");
    bench(1365,4096,1);
    bench(4096,3072,1);
    bench(4096,2048,1);
    bench(4096,1024,1);
    std::printf("\n--- several rows per work-group (idle lanes reused) ---\n");
    bench(4096,3072,2);
    bench(4096,1024,4);
    bench(4096,1024,8);
    return 0;
}