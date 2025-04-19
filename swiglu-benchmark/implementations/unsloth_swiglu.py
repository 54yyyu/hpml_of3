"""
Unsloth implementation of SwiGLU.

This implementation uses Triton kernels for optimized GPU computation
as provided by the Unsloth library.
"""

import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F


def torch_cuda_device(device):
    """Context manager to ensure operations use the correct CUDA device."""
    class DeviceContext:
        def __init__(self, device):
            self.device = device
            self.previous_device = None
        
        def __enter__(self):
            if self.device.type == 'cuda':
                self.previous_device = torch.cuda.current_device()
                torch.cuda.set_device(self.device)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.previous_device is not None:
                torch.cuda.set_device(self.previous_device)
    
    return DeviceContext(device)


def calculate_settings(n_cols):
    """Calculate optimal block size for the kernel."""
    # For Unsloth, we use a fixed block size of 1024
    return 1024


@triton.jit
def fg_kernel(e, g, h, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for SwiGLU forward pass (Unsloth version)."""
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)  # Convert back to original dtype
    # h = f * g
    h_row = f_row * g_row
    # Store h
    tl.store(h + offsets, h_row, mask=mask)


def swiglu_fg_kernel(e, g):
    """Apply the SwiGLU forward kernel.
    
    Args:
        e: Gate activation tensor
        g: Linear projection tensor
        
    Returns:
        Output tensor after applying SwiGLU
    """
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_cuda_device(e.device):
        fg_kernel[grid](e.reshape(-1), g.reshape(-1), h.reshape(-1), n_elements, BLOCK_SIZE=1024)
    return h


@triton.jit
def DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for SwiGLU backward pass (Unsloth version)."""
    block_idx = tl.program_id(0)
    offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    # e = e.float()
    # se = 1.0 / (1.0 + torch.exp(-e))
    se_row = tl.sigmoid(e_row)
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    # h = f * g
    h_row = f_row * g_row
    # df = DW * f
    df_row = DW_row * f_row
    # dg = DW * g
    dg_row = DW_row * g_row
    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)
    # Store derivatives in buffers
    tl.store(DW + offsets, h_row, mask=mask)  # h = f * g
    tl.store(e + offsets, df_row, mask=mask)  # df = DW * f
    tl.store(g + offsets, de_row, mask=mask)  # de


def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    """Apply the SwiGLU backward kernel.
    
    Args:
        DW: Gradient tensor
        e: Gate activation tensor
        g: Linear projection tensor
        
    Returns:
        Tuple of (output, gate_grad, proj_grad)
    """
    n_elements = e.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch_cuda_device(e.device):
        DWf_DW_dfg_kernel[grid](
            DW.reshape(-1), 
            e.reshape(-1), 
            g.reshape(-1), 
            n_elements, 
            BLOCK_SIZE=1024
        )
    return DW, e, g


class UnslothSwiGLUFunction(torch.autograd.Function):
    """PyTorch autograd function for SwiGLU using Unsloth kernels."""
    
    @staticmethod
    def forward(ctx, gate, proj):
        # Save tensors for backward pass
        ctx.save_for_backward(gate, proj)
        # Ensure tensors are contiguous
        gate = gate.contiguous()
        proj = proj.contiguous()
        # Apply SwiGLU
        return swiglu_fg_kernel(gate, proj)
    
    @staticmethod
    def backward(ctx, grad_output):
        gate, proj = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        # Create clones to avoid modifying the saved tensors
        gate_clone = gate.clone()
        proj_clone = proj.clone()
        grad_clone = grad_output.clone()
        
        # Apply backward kernel
        # The kernel modifies the tensors in-place and returns:
        # - grad_output: Contains the output (f * g)
        # - gate_clone: Contains df = grad_output * f
        # - proj_clone: Contains de
        _, dgate, dproj = swiglu_DWf_DW_dfg_kernel(grad_clone, gate_clone, proj_clone)
        
        return dproj, dgate


class UnslothSwiGLU(nn.Module):
    """SwiGLU module using Unsloth kernels.
    
    This implementation uses Triton kernels for optimized GPU computation.
    """
    
    def __init__(self, input_dim, output_dim):
        """Initialize the Unsloth SwiGLU module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.w_gate = nn.Linear(input_dim, output_dim, bias=False)
        self.w_proj = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        """Forward pass of Unsloth SwiGLU.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        gate = self.w_gate(x)
        proj = self.w_proj(x)
        
        # Handle cases where Triton kernels might not work
        if not torch.cuda.is_available():
            return F.silu(gate) * proj
            
        try:
            # Use custom autograd function for SwiGLU
            return UnslothSwiGLUFunction.apply(gate, proj)
        except Exception as e:
            # Fallback to standard PyTorch implementation if Triton fails
            print(f"Triton kernel failed, falling back to PyTorch: {e}")
            return F.silu(gate) * proj