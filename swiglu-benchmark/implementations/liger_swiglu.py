"""
Liger kernel implementation of SwiGLU.

This implementation uses Triton kernels for optimized GPU computation
as provided by the Liger library.
"""

import torch
import triton
import triton.language as tl
import torch.nn as nn


def calculate_settings(n_cols):
    """Calculate block size and number of warps for the kernel.
    
    Args:
        n_cols: Number of columns in the input tensor
        
    Returns:
        Tuple of (block_size, num_warps)
    """
    # Calculate the next power of 2 for the block size
    block_size = 2**int(torch.log2(torch.tensor(n_cols)).ceil().item())
    block_size = max(min(block_size, 2048), 32)
    
    # Calculate the number of warps based on block size
    num_warps = max(1, block_size // 256)
    return block_size, num_warps


def ensure_contiguous(func):
    """Decorator to ensure tensors are contiguous."""
    def wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.contiguous()
            new_args.append(arg)
        return func(*new_args, **kwargs)
    return wrapper


@triton.jit
def silu(x):
    """SiLU activation function: x * sigmoid(x)."""
    return x * tl.sigmoid(x)


@triton.jit
def swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for SwiGLU forward pass."""
    program_id = tl.program_id(0).to(tl.int64)
    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for SwiGLU backward pass."""
    program_id = tl.program_id(0).to(tl.int64)
    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_forward(a, b):
    """Forward pass for SwiGLU.
    
    Args:
        a: Gate input tensor
        b: Projection input tensor
        
    Returns:
        Tuple of (gate, projection, output)
    """
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):
    """Backward pass for SwiGLU.
    
    Args:
        a: Gate input tensor
        b: Projection input tensor
        dc: Gradient tensor
        
    Returns:
        Tuple of (gate_grad, projection_grad)
    """
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.view(*ori_shape), b.view(*ori_shape)


class LigerSiLUMulFunction(torch.autograd.Function):
    """PyTorch autograd function for SwiGLU using Liger kernels."""
    
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c
    
    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        # Use clones to avoid modifying saved tensors
        a_clone = a.clone()
        b_clone = b.clone()
        da, db = swiglu_backward(a_clone, b_clone, dc)
        return da, db


class LigerSwiGLU(nn.Module):
    """SwiGLU module using Liger kernels.
    
    This implementation uses Triton kernels for optimized GPU computation.
    """
    
    def __init__(self, input_dim, output_dim):
        """Initialize the Liger SwiGLU module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.w_gate = nn.Linear(input_dim, output_dim, bias=False)
        self.w_proj = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        """Forward pass of Liger SwiGLU.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        gate = self.w_gate(x)
        proj = self.w_proj(x)
        
        # Use the custom autograd function for SwiGLU
        return LigerSiLUMulFunction.apply(gate, proj)