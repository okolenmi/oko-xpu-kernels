/* NEGATIVE RESULT -- kept as evidence.
 * int8 pack strategies (scalar-from-float vs vector int32) and
 * store widths (16 B vs 64 B).
 * RESULT: no measurable difference on either axis.
 */
// The int8 store costs 0.215 ms = 40% of the rotate kernel for only 12.6 MB.
// Two suspects: (a) the scalar pack loop, (b) 16-byte stores at 256-byte lane
// stride -> each store instruction touches 16 partial 64B sectors.
// Measure pack variants x store widths, and the analogous question for loads.
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

constexpr int NREG=16;

static inline F16 quant(F16 v,float inv){ return sycl::min(sycl::max(sycl::rint(v*inv),F16(-127.f)),F16(127.f)); }

// --- pack variants: 16 floats -> 4 packed uint32 ---
static inline vec<uint32_t,4> pk_scalar(F16 v, float inv){
    const F16 r=quant(v,inv);
    vec<uint32_t,4> b;
    for(int t=0;t<16;t++) b[t>>2] |= (uint32_t)(uint8_t)(int8_t)(int32_t)r[t] << (8*(t&3));
    return b;
}
static inline vec<uint32_t,4> pk_i32(F16 v, float inv){
    const I32 i = quant(v,inv).convert<int32_t>();
    vec<uint32_t,4> b;
    for(int j=0;j<4;j++){
        b[j] = ((uint32_t)i[4*j] & 0xFFu) | (((uint32_t)i[4*j+1] & 0xFFu)<<8)
             | (((uint32_t)i[4*j+2] & 0xFFu)<<16) | (((uint32_t)i[4*j+3] & 0xFFu)<<24);
    }
    return b;
}
// arithmetic pack: clamp in int32, mask, then two shift/or "adds" over a
// strided view using swizzle (no per-lane scalar lanes)
static inline vec<uint32_t,4> pk_swz(F16 v, float inv){
    const I32 i = quant(v,inv).convert<int32_t>();
    // lane pairs: t and t+4 -> contributes bits at 0 and 4 -> shift by 8*(t&3)
    // build per-uint32 by interleaving: take even/odd halves
    vec<int32_t,16> lo = i.swizzle<0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15>();
    vec<int32_t,16> hi = i.swizzle<0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15>();
    // desired: word j = i[4j] | i[4j+1]<<8 | i[4j+2]<<16 | i[4j+3]<<24
    // gather the 4 quads as 4 int32 lanes is not expressible; do it with 2 halves
    vec<int32_t,16> a = i.swizzle<0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15>();
    vec<int32_t,16> b = i.swizzle<0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15>();
    (void)lo;(void)hi;(void)a;(void)b;
    return pk_i32(v,inv);
}

// --- store widths ---
static inline void st16(int8_t*p, F16 v, float inv){ vec<uint32_t,4> b=pk_i32(v,inv); std::memcpy(p,&b,16); }
static inline void st64(int8_t*p, F16 a,F16 b,F16 c,F16 d,float inv){
    vec<uint32_t,16> o;
    vec<uint32_t,4> t0=pk_i32(a,inv),t1=pk_i32(b,inv),t2=pk_i32(c,inv),t3=pk_i32(d,inv);
    for(int j=0;j<4;j++){ o[j]=t0[j]; o[4+j]=t1[j]; o[8+j]=t2[j]; o[12+j]=t3[j]; }
    std::memcpy(p,&o,64);
}

// --- loads ---
static inline U16 even_mask(){U16 m;for(int t=0;t<16;t++)m[t]=(t&1)?0u:0xFFFFFFFFu;return m;}
static inline F16 ld_cur(const uint16_t*p){
    const vec<uint32_t,8> u=reinterpret_cast<const vec<uint32_t,8>*>(p)[0];
    U16 w;for(int j=0;j<8;j++){w[2*j]=u[j]<<16;w[2*j+1]=u[j]&0xFFFF0000u;}
    return sycl::bit_cast<F16>(w);
}
static inline void ld64(const uint16_t*p, F16&a, F16&b){
    const vec<uint32_t,16> u=reinterpret_cast<const vec<uint32_t,16>*>(p)[0]; // 64 B
    U16 lo = u.swizzle<0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7>();
    U16 hi = u.swizzle<0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7>();
    // element 2j of lo = u[j], element 2j+1 = u[j+8]  -> wrong pairing
    a = sycl::bit_cast<F16>((u.swizzle<0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7>()<<16)&even_mask());
    b = sycl::bit_cast<F16>((u.swizzle<8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15>()<<16)&even_mask());
    (void)lo;(void)hi;
}

int main(int argc,char**argv){
    size_t M=argc>1?strtoul(argv[1],0,10):4096, K=argc>2?strtoul(argv[2],0,10):3072;
    size_t N=M*K;
    queue q;
    uint16_t*x=malloc_shared<uint16_t>(N,q);
    int8_t*o=malloc_shared<int8_t>(N,q);
    for(size_t i=0;i<N;i++)x[i]=0x3F80;
    auto bench=[&](const char*nm,auto fn){
        for(int i=0;i<5;i++) fn();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) fn();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
        std::printf("%-44s %7.4f ms\n",nm,ms); return ms;
    };
    std::printf("M=%zu K=%zu\n\n",M,K);

    // LOAD-only (sink into a per-lane float so nothing is dead)
    bench("load 32B x16 (production)",[&]{
        q.submit([&](handler&h){h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1>it){
            size_t row=it.get_global_id(0)>>4,lane=it.get_local_id(0)&15;
            if(lane*256>=K) return;
            const uint16_t*p=x+row*K+lane*256; float s=0;
            for(int w=0;w<NREG;w++) s+=ld_cur(p+w*16)[0];
            if(s==12345.f) o[row]=1;
        });}).wait();
    });
    bench("load 64B x8",[&]{
        q.submit([&](handler&h){h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1>it){
            size_t row=it.get_global_id(0)>>4,lane=it.get_local_id(0)&15;
            if(lane*256>=K) return;
            const uint16_t*p=x+row*K+lane*256; float s=0;
            for(int b=0;b<NREG;b+=2){ F16 A,B; ld64(p+b*16,A,B); s+=A[0]+B[0]; }
            if(s==12345.f) o[row]=1;
        });}).wait();
    });

    // STORE-only: 16 registers of constant data, no loads
    bench("store 16B x16 pack=scalar (production)",[&]{
        q.submit([&](handler&h){h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1>it){
            size_t row=it.get_global_id(0)>>4,lane=it.get_local_id(0)&15;
            if(lane*256>=K) return;
            int8_t*p=o+row*K+lane*256; F16 v((float)lane);
            for(int w=0;w<NREG;w++){ vec<uint32_t,4> b=pk_scalar(v,100.f); std::memcpy(p+w*16,&b,16);} }
        );}).wait();
    });
    bench("store 16B x16 pack=int32",[&]{
        q.submit([&](handler&h){h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1>it){
            size_t row=it.get_global_id(0)>>4,lane=it.get_local_id(0)&15;
            if(lane*256>=K) return;
            int8_t*p=o+row*K+lane*256; F16 v((float)lane);
            for(int w=0;w<NREG;w++) st16(p+w*16,v,100.f); }
        );}).wait();
    });
    bench("store 64B x4  pack=int32",[&]{
        q.submit([&](handler&h){h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1>it){
            size_t row=it.get_global_id(0)>>4,lane=it.get_local_id(0)&15;
            if(lane*256>=K) return;
            int8_t*p=o+row*K+lane*256; F16 v((float)lane);
            for(int b=0;b<NREG;b+=4) st64(p+b*16,v,v,v,v,100.f); }
        );}).wait();
    });
    return 0;
}