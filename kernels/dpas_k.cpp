#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
int main() {
  queue q(gpu_selector_v);
  int *out = malloc_shared<int>(4,q);
  q.submit([&](handler& h){
    h.parallel_for(range<1>(1), [=](id<1>){
      simd<signed char,16> a(1);
      simd<signed char,128> b(1);
      simd<int,8> c(0);
      c = dpas<8,1,int,signed char,signed char,dpas_argument_type::s8,dpas_argument_type::s8,128,16>(b,a);
      out[0]=c[0];
    });
  }).wait();
  std::cout << "k16 result=" << out[0] << "\n";
  sycl::free(out,q);
  return 0;
}
