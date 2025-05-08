# fused_swiglu_layer.py
import torch
import triton
import torch.nn.functional as F
from torch import nn, autograd

# our implementation
from fused_forward_kernel import swiglu_fused_forward_kernel
from fused_backward_kernel import swiglu_fused_backward_kernel
from fused_weight_grad_kernel import swiglu_fused_weight_grad_kernel

class _FusedSwiGLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, Wf, block):
        B = x.shape[0] # batch_size
        D = Wf.shape[1] // 2 # hidden_dim

        # allocate output
        z = torch.empty(B, D, device=x.device, dtype=x.dtype)
        # save for backward
        ctx.save_for_backward(x, Wf)
        ctx.block = block

        # launch Triton forward
        grid = (triton.cdiv(B, block), triton.cdiv(D, block))
        swiglu_fused_forward_kernel[grid](
            x, Wf, z,
            B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            z.stride(0),   z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        return z

    # HERE IS THE ONE WITH dWf, which 
    @staticmethod
    def backward(ctx, dZ):
        import torch, triton
        from fused_backward_kernel import swiglu_fused_backward_kernel
        from fused_weight_grad_kernel import swiglu_fused_weight_grad_kernel

        # recover saved inputs
        x, Wf = ctx.saved_tensors
        block = ctx.block
        B = x.shape[0]              # batch size
        D = Wf.shape[1] // 2        # hidden dim

        # 1) fused dX
        dX = torch.empty_like(x)
        grid2d = (triton.cdiv(B, block), triton.cdiv(D, block))
        swiglu_fused_backward_kernel[grid2d](
            x,   Wf, dZ, dX,
            B, D,
            x.stride(0),   x.stride(1),
            Wf.stride(0),  Wf.stride(1),
            dZ.stride(0),  dZ.stride(1),
            dX.stride(0),  dX.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )

        # 2) fused weight-gradient
        I = x.shape[1]              # input feature dim
        dWf = torch.empty(I, 2*D, device=x.device, dtype=x.dtype)
        # choose tile sizes for (B, I, D) dimensions
        BLOCK_B, BLOCK_I, BLOCK_D = 64, 64, 64
        grid3d = (
            triton.cdiv(B, BLOCK_B),
            triton.cdiv(I, BLOCK_I),
            triton.cdiv(D, BLOCK_D)
        )
        swiglu_fused_weight_grad_kernel[grid3d](
            x, dZ, dWf,
            B, I, D,
            x.stride(0),    x.stride(1),
            dZ.stride(0),   dZ.stride(1),
            dWf.stride(0),  dWf.stride(1),
            BLOCK_B=BLOCK_B, BLOCK_I=BLOCK_I, BLOCK_D=BLOCK_D
        )

        # return gradients for x, Wf, and None for block
        return dX, dWf, None

class FusedSwiGLU(nn.Module):
    """
    Fused SwiGLU forward+backward in one Triton kernel each.
    """
    def __init__(self, in_features: int, hidden_dim: int, block: int = 64):
        super().__init__()
        self.Wf = nn.Parameter(torch.randn(in_features, 2 * hidden_dim))
        self.hidden_dim = hidden_dim
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FusedSwiGLUFunction.apply(x, self.Wf, self.block)
