"""Reference PyTorch implementation of ConvRot INT8 linear."""
import torch


def convrot_int8_linear(x, weight, weight_scale, bias=None, group_size=256):
    """ConvRot-style INT8 linear using PyTorch primitives.

    This is a correctness reference, not an optimized kernel.
    """
    n_groups = x.shape[-1] // group_size
    h = torch.eye(group_size, device=x.device, dtype=x.dtype)
    # A real implementation would use a normalized Hadamard matrix here.
    x_rot = x.reshape(-1, n_groups, group_size) @ h
    x_rot = x_rot.reshape(x.shape)
    abs_max = x_rot.abs().amax(dim=-1, keepdim=True)
    x_scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    x_int8 = (x_rot.float() / x_scale).round().clamp(-127, 127).to(torch.int8)
    result = torch._int_mm(x_int8.reshape(-1, x.shape[-1]), weight.t().contiguous())
    result = (result.float() * x_scale) * weight_scale.view(1, -1)
    result = result.to(x.dtype)
    if bias is not None:
        result = result + bias
    return result.reshape(*x.shape[:-1], weight.shape[0])
