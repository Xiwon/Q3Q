import torch
import triton
import triton.language as tl
import os
from triton.runtime import driver

BLOCK_SIZE = 16
# lets just set the BLOCK_SIZE fixed now...

@triton.jit
def _rms_norm_gated_kernel(
    x_ptr, o_ptr, w_ptr, g_ptr,
    EPS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    id = tl.program_id(0)
    offset_n = id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_m = tl.arange(0, HEAD_DIM)
    x_block_ptr = x_ptr + (offset_n[:, None] * HEAD_DIM + offset_m[None, :])
    o_block_ptr = o_ptr + (offset_n[:, None] * HEAD_DIM + offset_m[None, :])
    g_block_ptr = g_ptr + (offset_n[:, None] * HEAD_DIM + offset_m[None, :])
    w_block_ptr = w_ptr + offset_m
    mask_n = offset_n < CHANNELS

    x = tl.load(x_block_ptr, mask=mask_n[:, None], other=0.).to(tl.float32) # (BLOCK_SIZE, HEAD_DIM)
    var = tl.sum(x * x, axis=1, keep_dims=True) / HEAD_DIM # (BLOCK_SIZE, 1)
    x = x * tl.rsqrt(var + EPS)
    w = tl.load(w_block_ptr).to(tl.float32) # (HEAD_DIM)
    out = x * w[None, :]

    # out = out * SiLU(gate)
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate = tl.load(g_block_ptr, mask=mask_n[:, None], other=0.).to(tl.float32)
    gate = gate / (1.0 + tl.exp(-gate))
    out = out * gate
    
    out = out.to(tl.bfloat16)
    tl.store(o_block_ptr, out, mask=mask_n[:, None])

def forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gate: torch.Tensor
) -> torch.Tensor:
    orig_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    gate = gate.reshape(-1, gate.shape[-1])
    channels, dim = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(channels, BLOCK_SIZE),)
    _rms_norm_gated_kernel[grid](x, out, weight, gate, eps, channels, dim, BLOCK_SIZE)
    return out.view(orig_shape)