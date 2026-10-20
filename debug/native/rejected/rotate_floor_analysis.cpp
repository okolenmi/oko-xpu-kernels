/* NEGATIVE RESULT -- kept as evidence.
 * Establishes the structural floor: the same copy loop costs
 * 0.402 ms with all 16 registers held live versus 0.128 ms with load and
 * store interleaved. This is where the 4x residency penalty was found.
 */
// Is the 0.35 ms "roundtrip floor" actually memory, or is it the scalar
// bf16->fp32 conversion / int8 packing?
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace sycl;

constexpr int GS = 256, NREG = 16;
using F16 = vec<float, 16>;

int main(int argc, char **argv) {
    size_t M = argc>1?strtoul(argv[1],0,10):4096;
    size_t K = argc>2?strtoul(argv[2],0,10):3072;
    size_t N = M*K;
    queue q;
    std::printf("%s  M=%zu K=%zu  (%.1f MB per pass)\n",
                q.get_device().get_info<info::device::name>().c_str(), M, K,
                (double)N*4/1e6);
    // float32 in, float32 out (pure vector copies, no conversion at all)
    float *fin = malloc_shared<float>(N,q), *fout = malloc_shared<float>(N,q);
    uint16_t *bin = malloc_shared<uint16_t>(N,q);
    int8_t *bout = malloc_shared<int8_t>(N,q);
    float *sc = malloc_shared<float>(M,q);
    for (size_t i=0;i<N;i++){ fin[i]=1.f; bin[i]=0x3F80; }

    auto bench=[&](const char*nm,auto fn,double bytes){
        for(int i=0;i<5;i++) fn();
        auto t0=std::chrono::high_resolution_clock::now();
        for(int i=0;i<20;i++) fn();
        auto t1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/20;
        std::printf("%-40s %.4f ms  %6.0f GB/s\n",nm,ms,bytes/1e9/(ms*1e-3));
    };

    // A: pure vector copy, register-resident (16 zmm held), float32
    auto A=[&]{
        q.submit([&](handler&h){ h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1> it){
            size_t row=it.get_global_id(0)>>4, lane=it.get_local_id(0)&15;
            if (row>=M || lane*GS>=K) return;
            F16 W[NREG]; const float*p=fin+row*K+lane*GS;
            for(int w=0;w<NREG;w++) W[w]=reinterpret_cast<const F16*>(p)[w];
            for(int w=0;w<NREG;w++) reinterpret_cast<F16*>(fout+row*K+lane*GS)[w]=W[w];
        });}).wait();
    };
    bench("A f32->f32 16zmm resident (8B/elem)", A, (double)N*8);

    // B: streaming copy, no register residency (load then immediately store per element)
    auto B=[&]{
        q.submit([&](handler&h){ h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1> it){
            size_t row=it.get_global_id(0)>>4, lane=it.get_local_id(0)&15;
            if (row>=M || lane*GS>=K) return;
            const float*p=fin+row*K+lane*GS; float*d=fout+row*K+lane*GS;
            for(int w=0;w<NREG;w++){ F16 v=reinterpret_cast<const F16*>(p)[w]; reinterpret_cast<F16*>(d)[w]=v; }
        });}).wait();
    };
    bench("B f32->f32 streaming", B, (double)N*8);

    // C: bf16 -> fp32 convert -> bf16 store (the real load path, no rotate)
    auto C=[&]{
        q.submit([&](handler&h){ h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1> it){
            size_t row=it.get_global_id(0)>>4, lane=it.get_local_id(0)&15;
            if (row>=M || lane*GS>=K) return;
            F16 W[NREG]; const uint16_t*p=bin+row*K+lane*GS;
            for(int w=0;w<NREG;w++){
                const vec<uint32_t,8> u=reinterpret_cast<const vec<uint32_t,8>*>(p)[w];
                vec<uint32_t,16> ww;
                for(int j=0;j<8;j++){ww[2*j]=u[j]<<16; ww[2*j+1]=u[j]&0xFFFF0000u;}
                W[w]=bit_cast<F16>(ww);
            }
            for(int w=0;w<NREG;w++) reinterpret_cast<F16*>(fout+row*K+lane*GS)[w]=W[w];
        });}).wait();
    };
    bench("C bf16->fp32->f32 (conversion only)", C, (double)N*6);

    // D: the real store path, fp32 in, int8 out, no rotate
    auto D=[&]{
        q.submit([&](handler&h){ h.parallel_for(nd_range<1>(range<1>(M*16),range<1>(16)),[=](nd_item<1> it){
            size_t row=it.get_global_id(0)>>4, lane=it.get_local_id(0)&15;
            if (row>=M || lane*GS>=K) return;
            F16 W[NREG]; const float*p=fin+row*K+lane*GS;
            for(int w=0;w<NREG;w++) W[w]=reinterpret_cast<const F16*>(p)[w];
            int8_t*d=bout+row*K+lane*GS;
            for(int w=0;w<NREG;w++){
                F16 r=min(max(rint(W[w]*100.f),F16(-127.f)),F16(127.f));
                uint32_t b[4]={0,0,0,0};
                for(int t=0;t<16;t++) b[t>>2]|=(uint32_t)(uint8_t)(int8_t)(int32_t)r[t]<<(8*(t&3));
                memcpy(d+w*16,b,16);
            }
        });}).wait();
    };
    bench("D f32->int8 store only", D, (double)N*5);
    return 0;
}