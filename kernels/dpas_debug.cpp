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
      simd<signed char,32> a(0);
      simd<signed char,256> b(1);
      for (int i=0;i<32;i++) a[i]=static_cast<signed char>(i+1);
      simd<int,8> c(0);
      c = dpas<8,1,int,signed char,signed char,dpas_argument_type::s8,dpas_argument_type::s8,256,32>(b,a);
      out[0]=c[0];
      // reset a all ones, b sequential
      for (int i=0;i<32;i++) a[i]=1;
      for (int i=0;i<256;i++) b[i]=static_cast<signed char>((i%32)+1);
      c = dpas<8,1,int,signed char,signed char,dpas_argument_type::s8,dpas_argument_type::s8,256,32>(b,a);
      out[1]=c[0];
    });
  }).wait();
  std::cout << "seqA_allB=" << out[0] << " allA_seqB=" << out[1] << "\n";
  sycl::free(out,q);
  return 0;
}
