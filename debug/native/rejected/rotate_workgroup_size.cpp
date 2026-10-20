/* NEGATIVE RESULT -- kept as evidence.
 * Work-group sizes 16/32/64/128/256.
 * RESULT: flat at 0.57-0.60 ms. Not a lever.
 */
// Experiment: does a larger work-group (more independent chains per work-group,
// same registers per lane) fix the latency-bound rotate?
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

// WG = work-group size (multiple of 16). Each 16-lane sub-group owns one row.
template <int WG>
static void launch(queue &q, const uint16_t *x, int8_t *o, float *sc, size_t M, size_t K) {
    const size_t ng = K / GS, total = M * 16;
    const size_t groups = (total + WG - 1) / WG;
    const F16 c0=h4c(0,0),c1=h4c(1,0),c2=h4c(2,0),c3=h4c(3,0);
    const F16 d0=h4c(0,1),d1=h4c(1,1),d2=h4c(2,1),d3=h4c(3,1);
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
            size_t gid = it.get_global_id(0);
            size_t lane = it.get_local_id(0) & 15;
            size_t row = gid >> 4;
            if (row >= M) return;
            const uint16_t *xr = x + row * K;
            F16 W[NREG];
            F16 amaxv(0.f);
            size_t grp = lane;
            if (grp < ng) {
                const uint16_t *p = xr + grp * GS;
                for (int w = 0; w < NREG; ++w) W[w] = ld(p + w*16);
                for (int w = 0; w < NREG; ++w)
                    W[w] = c0*W[w].swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>() +
                           c1*W[w].swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>() +
                           c2*W[w].swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>() +
                           c3*W[w].swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
                for (int w = 0; w < NREG; ++w)
                    W[w] = d0*W[w].swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>() +
                           d1*W[w].swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>() +
                           d2*W[w].swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>() +
                           d3*W[w].swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
                for (int g=0;g<4;g++) bfly4(W[4*g],W[4*g+1],W[4*g+2],W[4*g+3]);
                for (int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                for (int w = 0; w < NREG; ++w) amaxv = max(amaxv, fabs(W[w]));
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
            if (grp < ng) {
                int8_t *d = o + row * K + grp * GS;
                for (int w = 0; w < NREG; ++w) st(d + w*16, W[w], 1.f/s);
            }
        });
    }).wait();
}

int main(int argc, char **argv) {
    size_t M = argc>1?strtoul(argv[1],0,10):4096;
    size_t K = argc>2?strtoul(argv[2],0,10):3072;
    size_t N = M*K;
    queue q;
    uint16_t *x = malloc_shared<uint16_t>(N,q);
    int8_t *o = malloc_shared<int8_t>(N,q);
    float *sc = malloc_shared<float>(M,q);
    for (size_t i=0;i<N;i++) x[i]=0x3F80;
    auto bench=[&](const char*nm,auto fn){
        for(int i=0;i<5;i++) fn();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) fn();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
        std::printf("%-14s %.4f ms  %6.0f GB/s\n",nm,ms,(double)N*3/1e9/(ms*1e-3));
    };
    std::printf("M=%zu K=%zu\n",M,K);
    bench("WG=16",[&]{ launch<16>(q,x,o,sc,M,K); });
    bench("WG=32",[&]{ launch<32>(q,x,o,sc,M,K); });
    bench("WG=64",[&]{ launch<64>(q,x,o,sc,M,K); });
    bench("WG=128",[&]{ launch<128>(q,x,o,sc,M,K); });
    bench("WG=256",[&]{ launch<256>(q,x,o,sc,M,K); });
    return 0;
}