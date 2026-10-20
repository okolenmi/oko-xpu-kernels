#include <sycl/sycl.hpp>
#include <iostream>
int main() {
  sycl::queue q(sycl::gpu_selector_v);
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  return 0;
}
