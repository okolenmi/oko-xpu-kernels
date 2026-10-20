"""Minimal repro: cost of torch.utils.deterministic.fill_uninitialized_memory.

torch.use_deterministic_algorithms(True) enables
torch.utils.deterministic.fill_uninitialized_memory (default True).  Every
uninitialised allocation (torch.empty / empty_like / empty_strided) is then
filled with a known value (NaN for float) on the device.  A large model has one
such allocation per temporary, so this becomes one fill kernel per temporary,
and the fill kernels serialise with real work.

    python debug/bench_fill.py
"""
import time

import torch

torch.use_deterministic_algorithms(True)

# (shape, count): a small activation-sized temp, a large one, and a tensor of
# roughly the size the model allocates its biggest intermediates at.
CASES = [((3072,), 2000), ((128, 3072), 1000), ((1, 8192, 3072), 10)]


def bench(dev, shape, n, fill):
    torch.utils.deterministic.fill_uninitialized_memory = fill
    sync = torch.xpu.synchronize if dev.type == "xpu" else (lambda: None)
    sync()
    t = time.perf_counter()
    keep = [torch.empty(*shape, device=dev) for _ in range(n)]
    sync()
    dt = time.perf_counter() - t
    filled = bool(torch.isnan(keep[-1].float()).all().item())
    return dt / n * 1e6, filled


def main():
    print(f"torch {torch.__version__}")
    print(f"deterministic algorithms  = {torch.are_deterministic_algorithms_enabled()}")
    print("fill_uninitialized_memory = "
          f"{torch.utils.deterministic.fill_uninitialized_memory}")
    print("(the fill only happens while deterministic algorithms are on)\n")

    devs = [torch.device("cpu")]
    if torch.xpu.is_available():
        devs.append(torch.device("xpu", 0))

    for dev in devs:
        for shape, n in CASES:
            on = min(bench(dev, shape, n, True)[0] for _ in range(2))
            off = min(bench(dev, shape, n, False)[0] for _ in range(2))
            nan = bench(dev, shape, n, True)[1]
            print(f"{str(dev):8s} empty{str(shape):17s} x{n:5d}   "
                  f"on {on:9.2f} us/alloc   off {off:9.2f} us/alloc   "
                  f"delta {on - off:+9.2f} us   all-NaN when on = {nan}")

    print("\nflag now:", torch.utils.deterministic.fill_uninitialized_memory)


if __name__ == "__main__":
    main()
