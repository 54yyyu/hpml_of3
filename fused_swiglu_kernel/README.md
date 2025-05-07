## Fused SwiGLU: Quick Start Guide

### 1. Purpose

Replace the standard “two linears + LigerSiLUMulFunction.apply” pattern with a single, high-throughput Triton kernel that computes both branches and the gate in one launch—no API changes to your model, just swap in a new layer.

### 2. Prerequisites

* **CUDA GPU** (compute capability ≥ 7.0)
* Python ≥ 3.8, PyTorch ≥ 1.13
* Install:

  ```bash
  pip install triton==2.0.0
  ```

### 3. Installation

Copy the two files into your project root:

* **`fused_kernel.py`** (defines `swiglu_fused_kernel`)
* **`fused_swiglu_layer.py`** (wrapper Layer class, see below)

### 4. API Overview

```python
# fused_kernel.py
@triton.jit
def swiglu_fused_kernel(
    X_ptr, W_ptr, Out_ptr,
    B: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xd,
    stride_wd, stride_w2d,
    stride_outb, stride_outd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    …  # (your fused kernel implementation)
```

```python
# fused_swiglu_layer.py
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
```

### 5. How to swap in your model

#### **Before** (Liger-style)

```python
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

class Net(nn.Module):
    def __init__(…):
        …
        self.fc1_a = nn.Linear(in_features, hidden_dim)
        self.fc1_b = nn.Linear(in_features, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        …
        a = self.fc1_a(x)                 # GEMM #1
        b = self.fc1_b(x)                 # GEMM #2
        z = LigerSiLUMulFunction.apply(a,b)
        return self.fc2(z)
```

#### **After** (Fused)

```python
from fused_swiglu_layer import FusedSwiGLU

class Net(nn.Module):
    def __init__(…):
        …
        # replaces fc1_a, fc1_b
        self.swi_glu = FusedSwiGLU(in_features, hidden_dim, block=64)
        self.fc2     = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        …
        z = self.swi_glu(x)                # one fused kernel
        return self.fc2(z)
```

### 6. Tuning & Troubleshooting

* **`block` size** (`BLOCK_M`, `BLOCK_N`): default 64 works on most GPUs.
* **Masking**: kernel safely handles “ragged” tiles at edges.
* **Shape consistency**: ensure `in_features == x.size(1)` and `hidden_dim == Wf.shape[1]//2`.
* **Device**: only works on CUDA; guard with `if x.is_cuda:` fallback to Liger if needed.
* **Shared-memory limits**: if you see “out-of-resource” adjust `block` down (e.g. to 32).

### 7. Validation

After swapping:

1. **Sanity check output**

   ```python
   z_ref = LigerSiLUMulFunction.apply(x@W1a, x@W1b)
   z_new = net.swi_glu(x)
   assert torch.allclose(z_ref, z_new, atol=1e-6)
   ```
2. **Benchmark** on your target shapes to confirm speedup.
