"""The fused NF4 dequantizer: the format and the contract.

The format is the standard one (bitsandbytes' ``create_normal_map()``): 16
normal-distribution quantiles, a per-block absmax (64 elements per block by
default) and two 4-bit indices per byte, the even index in the low nibble.

What can be checked without a device is the *format* -- the codebook's
definition, the nibble order, the block layout -- because those are what a
storage format is: a byte-for-byte agreement with the reference decides
whether a weight stored by one implementation can be read by the other.

The kernel itself needs a device (it is Triton), and the device path is
checked by ``python -m xpu_kernels`` and the probes.
"""

import pytest
import torch

from xpu_kernels.triton import nf4

needs_a_device = pytest.mark.skipif(
    not (torch.xpu.is_available() or torch.cuda.is_available())
    or not nf4.enabled(),
    reason="the nf4 dequantizer is a Triton kernel: it needs a device and Triton")


def _packed(values):
    """Pack 4-bit indices the way the format does: the even index in the low
    nibble, two per byte."""
    v = torch.as_tensor(values, dtype=torch.uint8)
    lo, hi = v[0::2], v[1::2]
    return (lo | (hi << 4)).to(torch.uint8)


# -- the format ------------------------------------------------------------

def test_the_codebook_is_the_normal_quantiles():
    cb = nf4.codebook()
    assert cb.numel() == 16
    assert cb.dtype == torch.float32
    assert torch.equal(cb, cb.sort().values)          # sorted
    assert float(cb.max()) == 1.0                      # scaled so the max is 1
    assert float(cb.min()) < 0.0
    # the definition: bitsandbytes' create_normal_map() with its defaults
    offset = 0.9677083
    v1 = torch.special.ndtri(
        torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    v3 = -torch.special.ndtri(
        torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    want = torch.cat([v1, torch.zeros(1, dtype=torch.float64), v3])
    want = want.sort().values
    want = want / want.max()
    assert torch.equal(cb, want.to(torch.float32))


def test_the_nibble_order_is_the_even_index_in_the_low_nibble():
    packed = _packed([1, 2, 3, 4])
    cb = nf4.codebook()
    absmax = torch.ones(1, dtype=torch.float32)
    got = nf4._reference_chain(packed, absmax, cb, blocksize=4)
    # index 0 -> low nibble of byte 0, index 1 -> high nibble of byte 0
    assert float(got[0]) == float(cb[1])
    assert float(got[1]) == float(cb[2])
    assert float(got[2]) == float(cb[3])
    assert float(got[3]) == float(cb[4])


def test_the_scale_is_per_block():
    packed = _packed([0] * 8)
    cb = nf4.codebook()
    absmax = torch.tensor([2.0, 8.0], dtype=torch.float32)
    got = nf4._reference_chain(packed, absmax, cb, blocksize=4)
    assert torch.allclose(got[:4], cb[0] * 2.0)
    assert torch.allclose(got[4:], cb[0] * 8.0)


# -- what needs a device ---------------------------------------------------

@needs_a_device
def test_the_kernel_is_bit_identical_to_the_chain():
    dev = "xpu"
    torch.manual_seed(0)
    n_blocks = 1024
    packed = torch.randint(0, 256, (n_blocks * 32,), device=dev, dtype=torch.uint8)
    absmax = torch.rand(n_blocks, device=dev) * 0.02 + 1e-4
    cb = nf4.codebook(dev)
    want = nf4._reference_chain(packed, absmax, cb)
    got = nf4.dequant_nf4(packed, absmax, cb, dtype=torch.float32)
    assert got.view(torch.int32).equal(want.view(torch.int32))
    # and at the production dtype
    want_bf16 = want.to(torch.bfloat16)
    got_bf16 = nf4.dequant_nf4(packed, absmax, cb)
    assert got_bf16.view(torch.int16).equal(want_bf16.view(torch.int16))


@needs_a_device
def test_the_gate_is_active():
    gate = nf4.gates()["nf4_dequant"]
    assert gate.ok(), gate.describe()


def test_the_argument_checks():
    # The argument checks come before the gate, so a caller's mistake is
    # reported as such whether or not the kernel is verified on this machine.
    with pytest.raises(TypeError):
        nf4.dequant_nf4(torch.zeros(4, dtype=torch.int8),
                        torch.ones(1, dtype=torch.float32))
    with pytest.raises(TypeError):
        nf4.dequant_nf4(torch.zeros(4, dtype=torch.uint8),
                        torch.ones(1, dtype=torch.float16))
    with pytest.raises(ValueError):
        nf4.dequant_nf4(torch.zeros(4, dtype=torch.uint8),
                        torch.ones(1, dtype=torch.float32), blocksize=63)
    # too few scales: 4 packed bytes are 8 elements, and at 4 per block that
    # is two blocks
    with pytest.raises(ValueError):
        nf4.dequant_nf4(torch.zeros(4, dtype=torch.uint8),
                        torch.ones(1, dtype=torch.float32), blocksize=4)
