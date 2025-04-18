import torch
import time
import wandb
from torch import nn
from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

# ─── 1. Initialize W&B ─────────────────────────────────────────────────────────
wandb.init(
    project="swiglu-performance",
    config={
        "batch": 32,
        "dim": 4096,
        "warmups": 5,
    }
)
cfg = wandb.config

# ─── 2. Synthetic input ─────────────────────────────────────────────────────────
batch, dim = cfg.batch, cfg.dim
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(batch, dim, device=device, requires_grad=True)
b = torch.randn(batch, dim, device=device, requires_grad=True)
grad_out = torch.randn_like(a)

# ─── 3. Warm‑up PyTorch SwiGLU ──────────────────────────────────────────────────
for _ in range(cfg.warmups):
    c = nn.functional.silu(a) * b
    loss = c.mean()
    loss.backward()
    torch.cuda.synchronize()
    a.grad.zero_(); b.grad.zero_()

# ─── 4. Time standard PyTorch SwiGLU ────────────────────────────────────────────
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
c = nn.functional.silu(a) * b
loss = c.mean(); loss.backward()
end.record()
torch.cuda.synchronize()

pytorch_ms = start.elapsed_time(end)
print("PyTorch SwiGLU: ", pytorch_ms, "ms")
wandb.log({"pytorch_swiglu_ms": pytorch_ms})

# ─── 5. Zero grads ───────────────────────────────────────────────────────────────
a.grad.zero_(); b.grad.zero_()

# ─── 6. Warm‑up Liger SwiGLU ─────────────────────────────────────────────────────
for _ in range(cfg.warmups):
    aa, bb, cc = swiglu_forward(a, b)
    da, db = swiglu_backward(a, b, grad_out)
    torch.cuda.synchronize()

# ─── 7. Time Liger SwiGLU ────────────────────────────────────────────────────────
start.record()
aa, bb, cc = swiglu_forward(a, b)
da, db = swiglu_backward(a, b, grad_out)
end.record()
torch.cuda.synchronize()

liger_ms = start.elapsed_time(end)
print("Liger SwiGLU:   ", liger_ms, "ms")
wandb.log({"liger_swiglu_ms": liger_ms})

# ─── 8. Compute Speedup & Finish ─────────────────────────────────────────────────
speedup = pytorch_ms / liger_ms if liger_ms > 0 else float('inf')
print(f"Speedup:         {speedup:.2f}×")
wandb.log({"speedup": speedup})

wandb.finish()

