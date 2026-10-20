#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
int main() {
  queue q(gpu_selector_v);
  int *out = malloc_shared<int>(512,q);
  q.submit([&](handler& h){
    h.parallel_for(range<1>(512), [=](id<1> wi){
      int idx = (int)wi[0];
      simd<signed char,32> a(1);
      simd<signed char,256> b(1);
      if (idx < 32) a[idx]=0;
      else b[idx-32]=0;
      simd<int,8> c(0);
      c = dpas<8,1,int,signed char,signed char,dpas_argument_type::s8,dpas_argument_type::s8,256,32>(b,a);
      out[idx]=c[0];
    });
  }).wait();
  for (int i=0;i<32;i++) if (out[i]!=16) std::cout << "A zero " << i << " -> " << out[i] << "\n";
  for (int i=0;i<256;i++) if (out[32+i]!=16) std::cout << "B zero " << i << " -> " << out[32+i] << "\n";
  std::cout << "done\n";
  sycl::free(out,q);
  return 0;
}
