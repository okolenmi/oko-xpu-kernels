/* NEGATIVE RESULT -- kept as evidence.
 * fp16 butterfly and alternative in-register formulations.
 * RESULT: fp16 is 2x SLOWER (1.14 vs 0.55 ms) -- there are no native
 * 512-bit fp16 vector ops on this path, and register pressure was never the
 * binding constraint anyway.
 */
// Variants of the in-register axes of the ConvRot rotate.
// V0: current production form (sign vectors, one expression per axis)
// V1: merged axis0+axis1 loop (only 4 sign vectors live at a time)
// V2: explicit accumulation chain (fewer live temporaries)
// V3: fp16 butterfly (halves the data registers: 8 zmm instead of 16)
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
using H16 = vec<sycl::half, 16>;

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
// fp16 version
static inline void bfly4h(H16 &a0,H16 &a1,H16 &a2,H16 &a3){
    H16 p=a0+a1,m=a0-a1,q=a2+a3,n=a2-a3;
    a0=(p+n)*sycl::half(0.5f); a1=(p-n)*sycl::half(0.5f);
    a2=(m+q)*sycl::half(0.5f); a3=(q-m)*sycl::half(0.5f);
}

template <int VARIANT>
static void launch(queue &q, const uint16_t *x, int8_t *o, float *sc, size_t M, size_t K) {
    const size_t ng = K / GS;
    const F16 c0=h4c(0,0),c1=h4c(1,0),c2=h4c(2,0),c3=h4c(3,0);
    const F16 d0=h4c(0,1),d1=h4c(1,1),d2=h4c(2,1),d3=h4c(3,1);
    auto toH = [](F16 v) {
        H16 r;
        for (int t = 0; t < 16; t++) r[t] = sycl::half(v[t]);
        return r;
    };
    const H16 h0=toH(c0),h1=toH(c1),h2=toH(c2),h3=toH(c3);
    const H16 k0=toH(d0),k1=toH(d1),k2=toH(d2),k3=toH(d3);
    q.submit([&](handler &h) {
        h.parallel_for(nd_range<1>(range<1>(M*16), range<1>(16)), [=](nd_item<1> it) {
            size_t row = it.get_global_id(0) >> 4;
            size_t lane = it.get_local_id(0) & 15;
            const uint16_t *xr = x + row * K;
            F16 W[NREG];
            H16 V[NREG];
            F16 amaxv(0.f);
            size_t grp = lane;
            if (grp < ng) {
                const uint16_t *p = xr + grp * GS;
                for (int w = 0; w < NREG; ++w) W[w] = ld(p + w*16);
                if (VARIANT == 3) {
                    for (int w = 0; w < NREG; ++w) { H16 t; for (int q2=0;q2<16;q2++) t[q2]=sycl::half(W[w][q2]); V[w]=t; }
                    // axes 2,3 first (elementwise), then axes 0,1
                    for (int g=0;g<4;g++) bfly4h(V[4*g],V[4*g+1],V[4*g+2],V[4*g+3]);
                    for (int c=0;c<4;c++) bfly4h(V[c],V[c+4],V[c+8],V[c+12]);
                    for (int w = 0; w < NREG; ++w)
                        V[w] = h0*V[w].swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>() +
                               h1*V[w].swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>() +
                               h2*V[w].swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>() +
                               h3*V[w].swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
                    for (int w = 0; w < NREG; ++w)
                        V[w] = k0*V[w].swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>() +
                               k1*V[w].swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>() +
                               k2*V[w].swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>() +
                               k3*V[w].swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
                    for (int w = 0; w < NREG; ++w) {
                        F16 f; for (int q2=0;q2<16;q2++) f[q2]=float(V[w][q2]);
                        amaxv = max(amaxv, fabs(f));
                        st(o + row*K + grp*GS + w*16, f, 1.f/127.f);
                    }
                } else {
                    if (VARIANT == 0) {
                        for (int w=0;w<NREG;w++)
                            W[w] = c0*W[w].swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>() +
                                   c1*W[w].swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>() +
                                   c2*W[w].swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>() +
                                   c3*W[w].swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
                        for (int w=0;w<NREG;w++)
                            W[w] = d0*W[w].swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>() +
                                   d1*W[w].swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>() +
                                   d2*W[w].swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>() +
                                   d3*W[w].swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
                    } else if (VARIANT == 1) {
                        // merged loop: axis0 then axis1 on the same register
                        for (int w=0;w<NREG;w++) {
                            F16 t = c0*W[w].swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>() +
                                    c1*W[w].swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>() +
                                    c2*W[w].swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>() +
                                    c3*W[w].swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
                            W[w] = d0*t.swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>() +
                                   d1*t.swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>() +
                                   d2*t.swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>() +
                                   d3*t.swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
                        }
                    } else {  // V2: explicit accumulation chain
                        for (int w=0;w<NREG;w++) {
                            F16 t = c0 * W[w].swizzle<0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12>();
                            t = t + c1 * W[w].swizzle<1,1,1,1,5,5,5,5,9,9,9,9,13,13,13,13>();
                            t = t + c2 * W[w].swizzle<2,2,2,2,6,6,6,6,10,10,10,10,14,14,14,14>();
                            t = t + c3 * W[w].swizzle<3,3,3,3,7,7,7,7,11,11,11,11,15,15,15,15>();
                            F16 u = d0 * t.swizzle<0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3>();
                            u = u + d1 * t.swizzle<4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7>();
                            u = u + d2 * t.swizzle<8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11>();
                            u = u + d3 * t.swizzle<12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15>();
                            W[w] = u;
                        }
                    }
                    for (int g=0;g<4;g++) bfly4(W[4*g],W[4*g+1],W[4*g+2],W[4*g+3]);
                    for (int c=0;c<4;c++) bfly4(W[c],W[c+4],W[c+8],W[c+12]);
                    for (int w=0;w<NREG;w++) amaxv = max(amaxv, fabs(W[w]));
                    for (int w=0;w<NREG;w++) st(o + row*K + grp*GS + w*16, W[w], 1.f/127.f);
                }
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
            if (VARIANT != 3 && grp < ng) {
                // rewrote with correct scale: redo quantize here
                int8_t *d = o + row*K + grp*GS;
                (void)d;
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
    auto bench=[&](const char*nm,auto fn){
        for(int i=0;i<5;i++) fn();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) fn();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
        std::printf("%-34s %.4f ms  %6.0f GB/s\n",nm,ms,(double)N*3/1e9/(ms*1e-3));
    };
    std::printf("M=%zu K=%zu\n",M,K);
    bench("V0 current (fp32, 16 zmm)",[&]{ launch<0>(q,x,o,sc,M,K); });
    bench("V1 merged axis0+1 loop",   [&]{ launch<1>(q,x,o,sc,M,K); });
    bench("V2 explicit accumulation", [&]{ launch<2>(q,x,o,sc,M,K); });
    bench("V3 fp16 butterfly (8 zmm)",[&]{ launch<3>(q,x,o,sc,M,K); });
    return 0;
}