/* NEGATIVE RESULT -- kept as evidence.
 * Rotate and stage bf16, reduce the row scale in a second
 * kernel, quantize in a third. Prediction was 0.27 ms.
 * RESULT: 0.75 ms. Pass 1 alone is 0.73 ms because the 16 registers are still
 * live across the transform; the extra 25 MB of staging is not free.
 */
// Correct 3-pass ConvRot rotate+quantize, with a checksum to defeat DCE.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <sycl/ext/oneapi/bfloat16.hpp>
using BF = sycl::ext::oneapi::bfloat16;
using BF16 = sycl::vec<BF, 16>;
using namespace sycl;

constexpr int GS = 256, NREG = 16;
using F16 = vec<float, 16>;
using U16 = vec<uint32_t, 16>;

static const float H4[4][4] = {{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
static inline F16 h4c(int c, bool d1) {
    F16 v; for (int t=0;t<16;t++) v[t]=0.5f*H4[d1?((t>>2)&3):(t&3)][c]; return v;
}
static inline F16 ldb(const uint16_t *p) {
    const vec<uint32_t,8> u = reinterpret_cast<const vec<uint32_t,8>*>(p)[0];
    U16 w; for (int j=0;j<8;j++){w[2*j]=u[j]<<16; w[2*j+1]=u[j]&0xFFFF0000u;}
    return bit_cast<F16>(w);
}
static inline uint16_t f2bf(float f) {
    uint32_t u; memcpy(&u,&f,4); u += 0x7FFFu + (u>>16); return (uint16_t)(u>>16);
}
static inline float bf2f(uint16_t h) {
    uint32_t u = (uint32_t)h<<16; float f; memcpy(&f,&u,4); return f;
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

// PASS 1: rotate, stage x_rot (bf16), emit per-group abs-max
static void pass1(queue &q, const uint16_t *x, uint16_t *xr, float *gamax,
                  size_t M, size_t K) {
    const size_t ng = K / GS, total = M * ng;
    const F16 c0=h4c(0,0),c1=h4c(1,0),c2=h4c(2,0),c3=h4c(3,0);
    const F16 d0=h4c(0,1),d1=h4c(1,1),d2=h4c(2,1),d3=h4c(3,1);
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(total), [=](id<1> idx) {
            size_t row = idx[0] / ng, grp = idx[0] - row * ng;
            const uint16_t *p = x + row * K + grp * GS;
            F16 W[NREG];
            // pipeline: load+inreg+axis2 four registers at a time
#pragma unroll
            for (int g = 0; g < 4; g++) {
                for (int j = 0; j < 4; j++) W[4*g+j] = ldb(p + (4*g+j)*16);
                for (int j = 0; j < 4; j++) W[4*g+j] = inreg(W[4*g+j], c0,c1,c2,c3,d0,d1,d2,d3);
                bfly4(W[4*g], W[4*g+1], W[4*g+2], W[4*g+3]);
            }
            for (int c = 0; c < 4; c++) bfly4(W[c], W[c+4], W[c+8], W[c+12]);

            BF16 *dst = reinterpret_cast<BF16*>(xr + row * K + grp * GS);
            F16 amaxv(0.f);
            for (int w = 0; w < NREG; w++) {
                amaxv = max(amaxv, fabs(W[w]));
                { BF16 b; for (int t=0;t<16;t++) b[t]=BF(W[w][t]); dst[w]=b; }
            }
            F16 m0=max(amaxv,amaxv.swizzle<8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7>());
            F16 m1=max(m0,m0.swizzle<4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11>());
            F16 m2=max(m1,m1.swizzle<2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13>());
            F16 m3=max(m2,m2.swizzle<1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14>());
            gamax[idx[0]] = m3[0];
        });
    }).wait();
}

// PASS 2: per-row scale from per-group abs-max
static void pass2(queue &q, const float *gamax, float *scale, size_t M, size_t ng) {
    q.submit([&](handler &h) {
        h.parallel_for(range<1>(M), [=](id<1> idx) {
            float m = 0.f;
            for (size_t g = 0; g < ng; g++) m = fmax(m, gamax[(size_t)idx[0]*ng + g]);
            scale[idx[0]] = fmax(m / 127.f, 1e-30f);
        });
    }).wait();
}

// PASS 3: quantize staged x_rot -> int8  (pure streaming)
static void pass3(queue &q, const uint16_t *xr, const float *scale, int8_t *o,
                  size_t M, size_t K, int32_t *chk) {
    q.submit([&](handler &h) {
        h.parallel_for(range<2>(M, K/16), [=](id<2> idx) {
            const vec<uint32_t,8> u = reinterpret_cast<const vec<uint32_t,8>*>(
                xr + (size_t)idx[0]*K + (size_t)idx[1]*16)[0];
            vec<uint32_t,16> w;
            for (int j=0;j<8;j++){ w[2*j]=u[j]<<16; w[2*j+1]=u[j]&0xFFFF0000u; }
            F16 v = bit_cast<F16>(w);
            F16 r = min(max(rint(v / scale[idx[0]]), F16(-127.f)), F16(127.f));
            uint32_t b[2] = {0,0};
            for (int t=0;t<16;t++) b[t>>3] |= (uint32_t)(uint8_t)(int8_t)(int32_t)r[t] << (8*(t&7));
            memcpy(o + (size_t)idx[0]*K + (size_t)idx[1]*16, b, 16);
            if (idx[1] == 0) atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>(*chk)
                                  .fetch_add((int32_t)b[0]);
        });
    }).wait();
}

int main(int argc,char**argv){
    size_t M=argc>1?strtoul(argv[1],0,10):4096, K=argc>2?strtoul(argv[2],0,10):3072;
    size_t N=M*K, ng=K/GS;
    queue q;
    uint16_t *x=malloc_shared<uint16_t>(N,q);
    uint16_t *xr=malloc_shared<uint16_t>(N,q);
    float *gamax=malloc_shared<float>(M*ng,q);
    float *scale=malloc_shared<float>(M,q);
    int8_t *o=malloc_shared<int8_t>(N,q);
    int32_t *chk=malloc_shared<int32_t>(1,q);
    uint64_t s=0x9E3779B97F4A7C15ull;
    for(size_t i=0;i<N;i++){ s=s*6364136223846793005ull+1442695040888963407ull;
        float v=(float)(int)((s>>33)%2001-1000)/1000.f*(float)((i%7)+1);
        uint32_t u; memcpy(&u,&v,4); u += 0x7FFFu+(u>>16); x[i]=(uint16_t)(u>>16); }

    // ---- correctness vs CPU Kronecker reference ----
    pass1(q,x,xr,gamax,M,K); pass2(q,gamax,scale,M,ng); pass3(q,xr,scale,o,M,K,chk);
    { // CPU reference on a sample of rows
        const size_t RS = std::min<size_t>(M, 8);
        int bad=0; double worst=0;
        std::vector<uint16_t> hx(N), hxr(N);
        for(size_t i=0;i<N;i++){ hx[i]=x[i]; hxr[i]=xr[i]; }
        for(size_t m=0;m<RS;m++){
            float amax=0;
            for(size_t g=0;g<ng;g++){
                const uint16_t*src=&hx[m*K+g*GS]; float*dst=new float[GS];
                for(int i=0;i<GS;i++){
                    int i0=i&3,i1=(i>>2)&3,i2=(i>>4)&3,i3=(i>>6)&3; float acc=0;
                    for(int j=0;j<GS;j++){ int j0=j&3,j1=(j>>2)&3,j2=(j>>4)&3,j3=(j>>6)&3;
                        acc += bf2f(src[j])*H4[i0][j0]*H4[i1][j1]*H4[i2][j2]*H4[i3][j3]; }
                    dst[i]=acc/16.f;
                }
                for(int i=0;i<GS;i++) amax=std::fmax(amax,std::fabs(dst[i]));
                for(int i=0;i<GS;i++)
                    worst=std::max(worst,(double)std::fabs(dst[i]-bf2f(hxr[m*K+g*GS+i])));
                delete[] dst;
            }
            float sr=std::fmax(amax/127.f,1e-30f);
            if(std::fabs(sr-scale[m])>sr*2e-2f) bad++;
        }
        std::printf("stage-fp32 max abs err vs CPU: %.4f   scale bad rows: %d/%zu\n",worst,bad,RS);
    }

    auto bench=[&](const char*nm,auto fn){
        for(int i=0;i<5;i++) fn();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) fn();
        auto t1=std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double,std::milli>(t1-t0).count()/20;
    };
    double a=bench("",[&]{ pass1(q,x,xr,gamax,M,K); });
    double b=bench("",[&]{ pass2(q,gamax,scale,M,ng); });
    double c=bench("",[&]{ pass3(q,xr,scale,o,M,K,chk); });
    std::printf("\npass1 rotate+stage   %.4f ms  (%.0f GB/s, %.1f MB)\n",a,
                (double)N*4/1e9/(a*1e-3), (double)N*4/1e6);
    std::printf("pass2 row scale      %.4f ms\n", b);
    std::printf("pass3 quantize       %.4f ms  (%.0f GB/s, %.1f MB)\n",c,
                (double)N*3/1e9/(c*1e-3), (double)N*3/1e6);
    std::printf("TOTAL                %.4f ms   (single-pass was 0.4705)\n",a+b+c);
    std::printf("checksum %d\n", chk[0]);
    return 0;
}