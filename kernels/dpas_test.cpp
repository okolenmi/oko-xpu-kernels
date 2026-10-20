#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
using namespace sycl;
using namespace sycl::ext::intel::experimental::matrix;
int main() {
  queue q(gpu_selector_v);
  std::cout << q.get_device().get_info<info::device::name>() << "\n";
  range<2> g(16,16);
  buffer<int8_t,1> a(16*16), b(16*16);
  buffer<int32_t,1> c(16*16);
  q.submit([&](handler& h){
    auto pa=a.get_accessor(h,read_only);
    auto pb=b.get_accessor(h,read_only);
    auto pc=c.get_accessor(h,write_only, no_init);
    h.parallel_for(nd_range<2>(g,{16,16}), [=](nd_item<2> it){
      sub_group sg=it.get_sub_group();
      joint_matrix<sub_group,int8_t,use::a,8,16,layout::row_major> ma;
      joint_matrix<sub_group,int8_t,use::b,16,16,layout::row_major> mb;
      joint_matrix<sub_group,int32_t,use::accumulator,8,16,layout::row_major> acc;
      joint_matrix_fill(sg,acc,0);
      joint_matrix_load(sg,ma,pa.get_pointer(),16);
      joint_matrix_load(sg,mb,pb.get_pointer(),16);
      joint_matrix_mad(sg,acc,ma,mb,acc);
      joint_matrix_store(sg,acc,pc.get_pointer(),16,layout::row_major);
    });
  }).wait();
  std::cout << "DPAS int8 test ok\n";
  return 0;
}
