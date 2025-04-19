"""
PyTorch reference implementation of SwiGLU.

A standard PyTorch implementation of SwiGLU (Swish-Gated Linear Unit)
for benchmarking purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PyTorchSwiGLU(nn.Module):
    """Standard PyTorch implementation of SwiGLU.
    
    This is a reference implementation using PyTorch's built-in operations
    without any custom CUDA kernels.
    """
    
    def __init__(self, input_dim, output_dim):
        """Initialize the SwiGLU module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.w_gate = nn.Linear(input_dim, output_dim, bias=False)
        self.w_proj = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        """Forward pass of SwiGLU.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        gate = self.w_gate(x)
        proj = self.w_proj(x)
        
        # SwiGLU activation: SiLU(gate) * proj
        # where SiLU(x) = x * sigmoid(x)
        return F.silu(gate) * proj


# Memory-optimized version that trades speed for memory
class MemoryOptimizedSwiGLU(nn.Module):
    """Memory-optimized implementation of SwiGLU.
    
    This implementation combines both linear projections into a single
    matrix multiplication to save memory, at the cost of some speed.
    """
    
    def __init__(self, input_dim, output_dim):
        """Initialize the memory-optimized SwiGLU module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        # Combined weight matrix for both gate and projection
        self.w_combined = nn.Linear(input_dim, 2 * output_dim, bias=False)
        self.output_dim = output_dim
        
    def forward(self, x):
        """Forward pass of memory-optimized SwiGLU.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        combined = self.w_combined(x)
        
        # Split the combined output into gate and projection
        gate, proj = torch.split(combined, self.output_dim, dim=-1)
        
        # SwiGLU activation: SiLU(gate) * proj
        return F.silu(gate) * proj


# Function-based implementation for benchmarking core operation
def swiglu_forward(x, w_gate, w_proj):
    """Functional implementation of SwiGLU forward pass.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, input_dim]
        w_gate: Gate weights of shape [input_dim, output_dim]
        w_proj: Projection weights of shape [input_dim, output_dim]
        
    Returns:
        Output tensor of shape [batch_size, seq_len, output_dim]
    """
    gate = F.linear(x, w_gate)
    proj = F.linear(x, w_proj)
    return F.silu(gate) * proj


# Memory-efficient implementation using activation checkpointing
class CheckpointedSwiGLU(nn.Module):
    """SwiGLU with activation checkpointing for memory efficiency.
    
    Uses PyTorch's checkpoint mechanism to trade computation for memory
    by recomputing activations during the backward pass.
    """
    
    def __init__(self, input_dim, output_dim):
        """Initialize the checkpointed SwiGLU module.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.w_gate = nn.Linear(input_dim, output_dim, bias=False)
        self.w_proj = nn.Linear(input_dim, output_dim, bias=False)
        
    def _swiglu_forward(self, x):
        """Internal forward function for checkpointing.
        
        Args:
            x: Input tensor
            
        Returns:
            SwiGLU output
        """
        gate = self.w_gate(x)
        proj = self.w_proj(x)
        return F.silu(gate) * proj
        
    def forward(self, x):
        """Forward pass with activation checkpointing.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # Fixed: correctly import the checkpoint function and
        # explicitly set use_reentrant=False
        return checkpoint(
            self._swiglu_forward, 
            x, 
            use_reentrant=False,
            preserve_rng_state=False
        )