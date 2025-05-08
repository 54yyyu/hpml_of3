# activation_bench_large.py
import time, torch, torch.nn.functional as F
from torch import nn
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

# ───────────────────────── Config ───────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256        # bigger batch
dim        = 8192       # much bigger hidden size
iters      = 100

# ───────────────────────── Modules ──────────────────────────
relu_fc1   = nn.Linear(dim, dim).to(device)
relu_fc2   = nn.Linear(dim, dim).to(device)
swiglu_fcA = nn.Linear(dim, dim).to(device)
swiglu_fcB = nn.Linear(dim, dim).to(device)

# Dummy input
x = torch.randn(batch_size, dim, device=device)

# Warm-up
for _ in range(10):
    _ = relu_fc2(F.relu(relu_fc1(x)))
    a = swiglu_fcA(x); b = swiglu_fcB(x)
    _ = LigerSiLUMulFunction.apply(a, b)

# ─────────────────────── ReLU Timing ────────────────────────
torch.cuda.synchronize() if device.type=="cuda" else None
t0 = time.time()
for _ in range(iters):
    y = relu_fc2(F.relu(relu_fc1(x)))
torch.cuda.synchronize() if device.type=="cuda" else None
relu_ms = (time.time() - t0) * 1000 / iters

# ────────────────────── SwiGLU Timing ───────────────────────
torch.cuda.synchronize() if device.type=="cuda" else None
t0 = time.time()
for _ in range(iters):
    a = swiglu_fcA(x)
    b = swiglu_fcB(x)
    y = LigerSiLUMulFunction.apply(a, b)
torch.cuda.synchronize() if device.type=="cuda" else None
swiglu_ms = (time.time() - t0) * 1000 / iters

# ───────────────────── Report ──────────────────────────────
print(f"Batch={batch_size}, Dim={dim}, iters={iters}")
print(f"  ReLU path:   {relu_ms:.3f} ms/op")
print(f"  SwiGLU path: {swiglu_ms:.3f} ms/op")
