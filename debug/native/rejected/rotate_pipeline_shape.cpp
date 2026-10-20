/* NEGATIVE RESULT -- kept as evidence.
 * Loading and transforming one axis-2 group at a time instead of
 * loading all 16 registers first. This one IS a win and is in the library
 * (0.63 -> 0.50 ms) because it lets load latency overlap the butterfly.
 */
// Restructure the rotate so that group-of-4 chunks are loaded and transformed
// as a pipeline, instead of loading all 16 registers then transforming.
// Mathematically identical (the four axes commute), but it lets the compiler
// overlap load latency with transform work.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace sycl;

constexpr int GS = 256, NREG = 16;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;

static const float H4[4][4] = {{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
static inline F16 h4c(int c, bool d1) {
    F16 v; for (int t=0;t<16;t++) v[t]=0.5f*H4[d1?((t>>2)&3):(t&3)][c]; return v;
}
static inline F16 ld(const uint16_t *p) {
    const vec<uint32_t,8> u = reinterpret_cast<const vec<uint32_t,8>*>(p)[0];
    U16 w; for (int j=0;j<8;j++){w[2*j]=u[j]<<16; w[2*j+1]=u[j]&0xFFFF0000u;}
    return bit_cast<F16>(w);
}
static inline void st(int8_t *p, F16 v, float inv) {
    F16 r = min(max(rint(v*inv), F16(-127.f)), F16(127.f));
    uint32_t b[4]={0,0,0,0};
    for (int t=0;t<16;t++) b[t>>2] |= (uint32_t)(uint8_t)(int8_t)(int32_t)r[t] << (8*(t&3));
    memcpy(p, b, 16);
}
static inline void bfly4(F16 &a0,F16 &a1,F16 &a2,F16 &a3){
    F16 p=a0+a1,m=a0-a1,q=a2+a3,n=a2-a3;
    a0=(p+n)*0.5f; a1=(p-n)*0.5f; a2=(m+q)*0.5f; a3=(q-m)*0.5f;
}

// per-register axes 0 and 1 on one register
static inline F16 inreg(F16 v, const F16 &c0,const F16 &c1,const F16 &c2,const F16 &c3,
                        const F16 &d0,const F16 &d1,const F16 &d2,const F16 &d3) {
    F16 t = c0 * v.swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>();
    t = t + c1 * v.swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>();
    t = t + c2 * v.swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>();
    t = t + c3 * v.swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
    F16 u = d0 * t.swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>();
    u = u + d1 * t.swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>();
    u = u + d2 * t.swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>();
    u = u + d3 * t.swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
    return u;
}

// V=0: load all 16, then transform (baseline)
// V=1: load+transform chunk of 4 at a time (pipelined), axis2 per chunk, axis3 at end
template <int V, int CH>
static void launch(queue &q, const uint16_t *x, int8_t *o, float *sc, size_t M, size_t K) {
    const size_t ng = K / GS;
    const F16 c0=h4c(0,0),c1=h4c(1,0),c2=h4c(2,0),c3=h4c(3,0);
    const F16 d0=h4c(0,1),d1=h4c(1,1),d2=h4c(2,1),d3=h4c(3,1);
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M*16), range<1>(16)), [=](nd_item<1> it) {
            size_t row = it.get_global_id(0) >> 4;
            size_t lane = it.get_local_id(0) & 15;
            const uint16_t *xr = x + row * K;
            F16 W[NREG];
            F16 amaxv(0.f);
            size_t grp = lane;
            if (grp < ng) {
                const uint16_t *p = xr + grp * GS;
                if (V == 0) {
                    for (int w=0;w<NREG;w++) W[w]=ld(p+w*16);
                    for (int w=0;w<NREG;w++) W[w]=inreg(W[w],c0,c1,c2,c3,d0,d1,d2,d3);
                    for (int g=0;g<4;g++) bfly4(W[4*g],W[4*g+1],W[4*g+2],W[4*g+3]);
                    for (int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                } else {
                    // CH chunk size (1,2,4): load+inreg+axis2 per chunk
                    for (int b=0; b<16; b+=CH) {
                        for (int j=0;j<CH;j++) W[b+j] = ld(p + (b+j)*16);
                        for (int j=0;j<CH;j++)
                            W[b+j] = inreg(W[b+j],c0,c1,c2,c3,d0,d1,d2,d3);
                        for (int g=0; g<CH/4; g++)
                            bfly4(W[b+4*g],W[b+4*g+1],W[b+4*g+2],W[b+4*g+3]);
                    }
                    for (int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                }
                for (int w=0;w<NREG;w++) amaxv = max(amaxv, fabs(W[w]));
                for (int w=0;w<NREG;w++) st(o + row*K + grp*GS + w*16, W[w], 1.f/127.f);
            }
            F16 r0=max(amaxv,amaxv.swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());
            F16 r1=max(r0,r0.swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>());
            F16 r2=max(r1,r1.swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>());
            F16 r3=max(r2,r2.swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>());
            auto sg = it.get_sub_group();
            float v = r3[0];
            v = fmax(v, select_from_group(sg, v, (uint32_t)(lane ^ 8)));
            v = fmax(v, select_from_group(sg, v, (uint32_t)(lane ^ 4)));
            v = fmax(v, select_from_group(sg, v, (uint32_t)(lane ^ 2)));
            v = fmax(v, select_from_group(sg, v, (uint32_t)(lane ^ 1)));
            float s = fmax(v/127.f, 1e-30f);
            if (lane == 0) sc[row] = s;
            if (grp < ng)
                for (int w=0;w<NREG;w++) st(o + row*K + grp*GS + w*16, W[w], 1.f/s);
        });
    }).wait();
}

int main(int argc,char**argv){
    size_t M=argc>1?strtoul(argv[1],0,10):4096, K=argc>2?strtoul(argv[2],0,10):3072;
    size_t N=M*K;
    queue q;
    uint16_t *x=malloc_shared<uint16_t>(N,q);
    int8_t *o=malloc_shared<int8_t>(N,q);
    float *sc=malloc_shared<float>(M,q);
    for(size_t i=0;i<N;i++) x[i]=0x3F80;
    auto bench=[&](const char*nm,auto fn){
        for(int i=0;i<5;i++) fn();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) fn();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
        std::printf("%-40s %.4f ms  %6.0f GB/s\n",nm,ms,(double)N*3/1e9/(ms*1e-3));
    };
    std::printf("M=%zu K=%zu\n",M,K);
    bench("V0 load-all then transform",[&]{ launch<0,4>(q,x,o,sc,M,K); });
    bench("V1 chunk=4",[&]{ launch<1,4>(q,x,o,sc,M,K); });
    bench("V1 chunk=2",[&]{ launch<1,2>(q,x,o,sc,M,K); });
    bench("V1 chunk=1",[&]{ launch<1,1>(q,x,o,sc,M,K); });
    return 0;
}