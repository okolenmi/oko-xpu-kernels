#include <sycl/sycl.hpp>
#include <iostream>
int main() {
  for (auto const& p : sycl::platform::get_platforms()) {
    std::cout << "platform: " << p.get_info<sycl::info::platform::name>() << "\n";
    for (auto const& d : p.get_devices()) {
      std::cout << "  device: " << d.get_info<sycl::info::device::name>() << "\n";
    }
  }
  return 0;
}
