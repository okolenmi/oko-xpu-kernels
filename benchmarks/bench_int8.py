"""Benchmark PyTorch INT8 matmul against BF16 on XPU."""
import time
import torch


def main():
    m, n, k = 4096, 3072, 3072
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    wbf = torch.randn(n, k, device="xpu", dtype=torch.bfloat16)
    xi = torch.randint(-127, 127, (m, k), device="xpu", dtype=torch.int8)
    wi = torch.randint(-127, 127, (n, k), device="xpu", dtype=torch.int8)

    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        _ = x @ wbf.t()
        torch.xpu.synchronize()
    bf16 = (time.perf_counter() - t) / 10 * 1000

    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        _ = torch._int_mm(xi, wi.t())
        torch.xpu.synchronize()
    intmm = (time.perf_counter() - t) / 10 * 1000

    print(f"BF16: {bf16:.3f} ms")
    print(f"INT8 _int_mm: {intmm:.3f} ms")
    print(f"speedup: {bf16 / intmm:.2f}x")


if __name__ == "__main__":
    main()
