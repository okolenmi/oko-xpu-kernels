/* NEGATIVE RESULT -- kept as evidence.
 * Stage the strided axis-3 butterfly through work-group memory
 * to cut peak registers from 16 zmm to ~4.
 * RESULT: 1.33 ms, 2.4x slower. Barriers and the shared round-trip cost far
 * more than the register pressure they save.
 */
// Shared-memory staged ConvRot rotate+quantize.
//
// The four Kronecker axes are separable, so the axis-2 groups
// ({4g..4g+3}) can be computed as soon as their 4 registers are loaded, and
// the result staged in work-group memory. Only then is the axis-3 butterfly
// applied, reading the staged values. Peak live registers drop from 16 zmm
// (1 KB/lane) to ~4, roughly tripling occupancy and letting load latency
// overlap with the transform.
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
static inline F16 inreg(F16 v, const F16 &c0,const F16 &c1,const F16 &c2,const F16 &c3,
                        const F16 &d0,const F16 &d1,const F16 &d2,const F16 &d3){
    F16 t = c0*v.swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>();
    t = t + c1*v.swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>();
    t = t + c2*v.swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>();
    t = t + c3*v.swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
    F16 u = d0*t.swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>();
    u = u + d1*t.swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>();
    u = u + d2*t.swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>();
    u = u + d3*t.swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
    return u;
}
static inline void bfly4(F16 &a0,F16 &a1,F16 &a2,F16 &a3){
    F16 p=a0+a1,m=a0-a1,q=a2+a3,n=a2-a3;
    a0=(p+n)*0.5f; a1=(p-n)*0.5f; a2=(m+q)*0.5f; a3=(q-m)*0.5f;
}
static inline void st(int8_t *p, F16 v, float inv) {
    F16 r = min(max(rint(v*inv), F16(-127.f)), F16(127.f));
    uint32_t b[4]={0,0,0,0};
    for (int t=0;t<16;t++) b[t>>2] |= (uint32_t)(uint8_t)(int8_t)(int32_t)r[t] << (8*(t&3));
    memcpy(p, b, 16);
}

static void launch(queue &q, const uint16_t *x, int8_t *o, float *sc, size_t M, size_t K) {
    const size_t ng = K / GS;
    const F16 c0=h4c(0,0),c1=h4c(1,0),c2=h4c(2,0),c3=h4c(3,0);
    const F16 d0=h4c(0,1),d1=h4c(1,1),d2=h4c(2,1),d3=h4c(3,1);
    q.submit([&](handler &h) {
        // 16 lanes x 2 x 256 floats = 32 KB per work-group
        local_accessor<float, 1> lx(range<1>(16 * 2 * GS), h);
        h.parallel_for(nd_range<1>(range<1>(M*16), range<1>(16)), [=](nd_item<1> it) {
            size_t row = it.get_global_id(0) >> 4;
            size_t lane = it.get_local_id(0) & 15;
            float *A = lx.get_pointer() + lane * 2 * GS;   // 512 floats
            float *B = A + GS;
            const uint16_t *xr = x + row * K;
            size_t grp = lane;
            bool active = (grp < ng) && (row < M);
            F16 amaxv(0.f);

            if (active) {
                const uint16_t *p = xr + grp * GS;
                // phase 1: per axis-2 group of 4 registers -> stage
#pragma unroll
                for (int g = 0; g < 4; g++) {
                    F16 R[4];
#pragma unroll
                    for (int j = 0; j < 4; j++) R[j] = ld(p + (4*g + j) * 16);
#pragma unroll
                    for (int j = 0; j < 4; j++) R[j] = inreg(R[j], c0,c1,c2,c3,d0,d1,d2,d3);
                    bfly4(R[0], R[1], R[2], R[3]);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        reinterpret_cast<F16*>(A + (4*g + j) * 16)[0] = R[j];
                }
            }
            it.get_sub_group().barrier();  // work-group size == sub-group size

            // phase 2: axis-3 butterfly reading staged values
            if (active) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    F16 R[4];
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        R[k] = reinterpret_cast<const F16*>(A + (c + 4*k) * 16)[0];
                    bfly4(R[0], R[1], R[2], R[3]);
                    amaxv = max(amaxv, fabs(R[0]));
                    amaxv = max(amaxv, fabs(R[1]));
                    amaxv = max(amaxv, fabs(R[2]));
                    amaxv = max(amaxv, fabs(R[3]));
#pragma unroll
                    for (int k = 0; k < 4; k++)
                        reinterpret_cast<F16*>(B + (c + 4*k) * 16)[0] = R[k];
                }
            }
            it.get_sub_group().barrier();  // work-group size == sub-group size

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
            float s = fmax(v / 127.f, 1e-30f);
            if (lane == 0) sc[row] = s;

            // phase 3: quantize from staged values
            if (active) {
                int8_t *d = o + row * K + grp * GS;
#pragma unroll
                for (int j = 0; j < 16; j++)
                    st(d + j * 16, reinterpret_cast<const F16*>(B + j * 16)[0], 1.f / s);
            }
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
    for(int i=0;i<5;i++) launch(q,x,o,sc,M,K);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<20;i++) launch(q,x,o,sc,M,K);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
    std::printf("shared-memory staged: %.4f ms  %6.0f GB/s  (was 0.4705 register-resident)\n",
                ms,(double)N*3/1e9/(ms*1e-3));
    return 0;
}