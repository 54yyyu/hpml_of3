## Fused SwiGLU: Quick Start Guide

### 1. Purpose

Provide a drop-in replacement for the "two linears + LigerSiLUMulFunction.apply" pattern with a single high-throughput Triton kernel that computes both branch projections and the GLU gate in one launch—and optionally fuses the input-gradient (dX) and/or weight-gradient (dW) as needed.

### 2. Prerequisites

- **CUDA GPU** (compute capability ≥ 7.0)  
- Python ≥ 3.8, PyTorch ≥ 1.13  
- Triton JIT/AOT compiler:  
```bash
pip install triton
```
- To run the tests in tests/:
```bash
# install these libraries
pip install torch torchvision
pip install liger-kernel
```

### 3. Installation

Copy the following into your project root:

* `fused_forward_kernel.py` (implements `swiglu_fused_forward_kernel`)
* `fused_backward_kernel.py` (implements `swiglu_fused_backward_kernel`)
* `fused_swiglu_layer.py` (wrapper module)
* *\[optional]* `fused_weight_grad_kernel.py` (for full weight-grad fusion)
* *\[optional]* `fused_swiglu_layer_WF.py` (wrapper with weight-grad module)

### 4. API Overview

#### `fused_forward_kernel.py`

```python
@triton.jit
def swiglu_fused_forward_kernel(
    X_ptr, Wf_ptr, Out_ptr,
    B: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xd,
    stride_wfb, stride_wfd,
    stride_ob, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ...  # forward GEMM + gate fused kernel
```

#### `fused_backward_kernel.py`

```python
@triton.jit
def swiglu_fused_backward_kernel(
    X_ptr, Wf_ptr, dZ_ptr, dX_ptr,
    B: tl.constexpr, D: tl.constexpr,
    ...,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ...  # input-gradient (dX) fused kernel
```

#### *\[optional]* `fused_weight_grad_kernel.py`

```python
@triton.jit
def swiglu_fused_weight_grad_kernel(
    X_ptr, dZ_ptr, dWf_ptr,
    B: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    ...,
    BLOCK_B: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_D: tl.constexpr
):
    ...  # weight-gradient (dWf) fused kernel
```

#### `fused_swiglu_layer.py`

```python
import torch, triton
from torch import nn, autograd
from fused_forward_kernel import swiglu_fused_forward_kernel
from fused_backward_kernel import swiglu_fused_backward_kernel
# optional import of fused_weight_grad_kernel if you need to learn Wf

class _FusedSwiGLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, Wf, block):
        B, D = x.shape[0], Wf.shape[1]//2
        z = torch.empty(B, D, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x, Wf)
        ctx.block = block
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

    @staticmethod
    def backward(ctx, dZ):
        import torch
        x, Wf = ctx.saved_tensors
        block = ctx.block
        B, D = x.shape[0], Wf.shape[1]//2

        # 1) fused input-grad
        dX = torch.empty_like(x)
        grid2d = (triton.cdiv(B, block), triton.cdiv(D, block))
        swiglu_fused_backward_kernel[grid2d](
            x, Wf, dZ, dX,
            B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            dZ.stride(0), dZ.stride(1),
            dX.stride(0), dX.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )

        # 2) weight-grad (optional)
        # if you want to train Wf, go to fused_swiglu_layer_WF.py
        # and tune the following:
        #
        # I = x.shape[1]
        # dWf = torch.empty(I, 2*D, device=x.device, dtype=x.dtype)
        # from fused_weight_grad_kernel import swiglu_fused_weight_grad_kernel
        # BLOCK_B, BLOCK_I, BLOCK_D = 64, 64, 64
        # grid3d = (
        #     triton.cdiv(B, BLOCK_B),
        #     triton.cdiv(I, BLOCK_I),
        #     triton.cdiv(D, BLOCK_D),
        # )
        # swiglu_fused_weight_grad_kernel[grid3d](
        #     x, dZ, dWf,
        #     B, I, D,
        #     x.stride(0), x.stride(1),
        #     dZ.stride(0), dZ.stride(1),
        #     dWf.stride(0), dWf.stride(1),
        #     BLOCK_B=BLOCK_B, BLOCK_I=BLOCK_I, BLOCK_D=BLOCK_D
        # )
        # return dX, dWf, None

        # default: no weight-grad -> maximum speedup
        return dX, None, None

class FusedSwiGLU(nn.Module):
    """
    Module wrapper: forward+input-grad fused, optional weight-grad.
    """
    def __init__(self, in_features, hidden_dim, block=64):
        super().__init__()
        self.Wf = nn.Parameter(torch.randn(in_features, 2*hidden_dim))
        self.hidden_dim = hidden_dim
        self.block = block

    def forward(self, x):
        return _FusedSwiGLUFunction.apply(x, self.Wf, self.block)
```

### 5. Swapping in Your Model

#### Before (Liger-style)

```python
self.fc1_a = nn.Linear(I, D)
self.fc1_b = nn.Linear(I, D)
…
a = self.fc1_a(x)
b = self.fc1_b(x)
z = LigerSiLUMulFunction.apply(a, b)
out = self.fc2(z)
```

#### After (Fused)

```python
from fused_swiglu_layer import FusedSwiGLU
…
self.fc1 = FusedSwiGLU(I, D, block=64)
self.fc2 = nn.Linear(D, out_features)
…
z = self.fc1(x)      # one fused kernel
out = self.fc2(z)
```

### 6. Weight-Grad Options & Performance

* **No weight-grad** (`return dX, None, None`):

  * Matches Liger's gate-only autograd
  * Keeps **2–3×** speedup on training & inference
  * `Wf` remains fixed (no learning)

* **Python weight-grad** (checkout `backward` in `fused_swiglu_layer_WF.py`):

  * Enables training `Wf` easily
  * Introduces two large GEMMs -> performance drops back near baseline

* **Triton weight-grad** (use `fused_weight_grad_kernel.py`):

  * Full fusion: forward, dX, dW in Triton
  * Requires 3D‐tiled kernel tuning
  * Can recover most speedups once tuned

### 7. Tuning & Troubleshooting

* **Block size**: try 32/64/128 for memory vs. occupancy
* **Masking**: handles edge tiles safely
* **Shape checks**: ensure `x.size(1)==in_features`, `Wf.shape[1]==2*hidden_dim`
* **CUDA guard**: fallback to Liger for CPU/MPS
* **Shared-memory errors**: reduce `block` if "out of resources"

### 8. Validation & Benchmark (more in tests/)

1. **Sanity check**

   ```python
   z_ref = LigerSiLUMulFunction.apply(x@W1a, x@W1b)
   z_new = fused_layer(x)
   assert torch.allclose(z_ref, z_new, atol=1e-6)
   ```
2. **Benchmarks**

   * Forward only: `mlp_forward_compare.py`
   * Backward only: `mlp_backward_compare.py`
   * Backward + WF: `mlp_backward_compare_WF.py`
   * End-to-end MNIST training: compare `liger_swiglu_main.py` vs. `fused_swiglu_main.py` vs. `fused_swiglu_main_WF.py`