#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
int main() {
  queue q(gpu_selector_v);
  int *out = malloc_shared<int>(16,q);
  q.submit([&](handler& h){
    h.parallel_for(range<1>(1), [=](id<1>){
      simd<signed char,32> a(1);
      simd<signed char,512> b(1);
      simd<int,16> c(0);
      c = dpas<8,1,int,signed char,signed char,dpas_argument_type::s8,dpas_argument_type::s8,512,32>(b,a);
      for(int i=0;i<16;i++) out[i]=c[i];
    });
  }).wait();
  std::cout << "exec16 result=" << out[0] << "\n";
  sycl::free(out,q);
  return 0;
}
