import time
import torch
import torch.nn.functional as F
from torch import nn
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

# ─────────────────────────── Config ────────────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size  = 64
dim         = 2048
iterations  = 200

# ────────────────────────── Modules ────────────────────────────
# ReLU path: Linear→ReLU→Linear
relu_fc1 = nn.Linear(dim, dim, bias=True).to(device)
relu_fc2 = nn.Linear(dim, dim, bias=True).to(device)

# SwiGLU path: Linear_A→Linear_B→SwiGLU
swiglu_fc_a = nn.Linear(dim, dim, bias=True).to(device)
swiglu_fc_b = nn.Linear(dim, dim, bias=True).to(device)

# identical input tensor
x = torch.randn(batch_size, dim, device=device)

# ────────────────────────── Warm-up ────────────────────────────
for _ in range(10):
    _ = relu_fc2(F.relu(relu_fc1(x)))
    a = swiglu_fc_a(x); b = swiglu_fc_b(x)
    _ = LigerSiLUMulFunction.apply(a, b)

# ───────────────────────── Timing ReLU ─────────────────────────
torch.cuda.synchronize() if device.type=="cuda" else None
t0 = time.time()
for _ in range(iterations):
    y = relu_fc2(F.relu(relu_fc1(x)))
torch.cuda.synchronize() if device.type=="cuda" else None
relu_time = (time.time() - t0) * 1000 / iterations

# ───────────────────────── Timing SwiGLU ───────────────────────
torch.cuda.synchronize() if device.type=="cuda" else None
t0 = time.time()
for _ in range(iterations):
    a = swiglu_fc_a(x)
    b = swiglu_fc_b(x)
    y = LigerSiLUMulFunction.apply(a, b)
torch.cuda.synchronize() if device.type=="cuda" else None
swiglu_time = (time.time() - t0) * 1000 / iterations

# ────────────────────── Report Results ─────────────────────────
print(f"Avg per-iteration over {iterations} runs:")
print(f"  ReLU path:   {relu_time:.3f} ms")
print(f"  SwiGLU path: {swiglu_time:.3f} ms")
