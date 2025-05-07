import torch, triton
import torch.nn.functional as F
from torch import nn
from fused_kernel import swiglu_fused_kernel

class FusedSwiGLU(nn.Module):
    """
    Replaces:
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
    with one Triton kernel launch:
        z = fused_GEMM_gate(x, Wf)
    """
    def __init__(self, in_features: int, hidden_dim: int, block: int = 64):
        super().__init__()
        # Wf packs [W1a | W1b] of shapes [in_features, hidden_dim]
        self.Wf = nn.Parameter(torch.randn(in_features, 2*hidden_dim))
        self.hidden_dim = hidden_dim
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _ = x.shape
        D = self.hidden_dim

        # allocate output
        z = torch.empty(B, D, device=x.device, dtype=x.dtype)

        # grid = (#tiles over B, #tiles over D)
        grid = (triton.cdiv(B, self.block), triton.cdiv(D, self.block))

        # launch fused kernel
        swiglu_fused_kernel[grid](
            x, self.Wf, z,
            B, D,
            x.stride(0), x.stride(1),
            self.Wf.stride(0), self.Wf.stride(1),
            z.stride(0),   z.stride(1),
            BLOCK_M=self.block, BLOCK_N=self.block
        )
        return z
