#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>
using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
int main() {
  queue q(gpu_selector_v);
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";
  int *out = malloc_shared<int>(8, q);
  q.submit([&](handler& h){
    h.parallel_for(range<1>(1), [=](id<1>){
      simd<signed char,256> b(1);
      simd<signed char,32> a(1);
      simd<int,8> c(0);
      c = dpas<8,1,int,signed char,signed char,sycl::ext::intel::esimd::xmx::dpas_argument_type::s8,sycl::ext::intel::esimd::xmx::dpas_argument_type::s8,256,32>(b,a);
      for(int i=0;i<8;i++) out[i]=c[i];
    });
  }).wait();
  std::cout << "dpas ok: " << out[0] << "\n";
  sycl::free(out, q);
  return 0;
}
