# activation_only_bench.py
import time, torch, torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 1024
dim        = 65536   # really large so we’re memory-bound
iters      = 200

# Single random vectors
a = torch.randn(batch_size, dim, device=device)
b = torch.randn(batch_size, dim, device=device)

# Warm-up
for _ in range(10):
    _ = F.relu(a)
    _ = LigerSiLUMulFunction.apply(a, b)

# Time ReLU
torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    out = F.relu(a)
torch.cuda.synchronize()
relu_ms = (time.time() - t0) * 1000 / iters

# Time SwiGLU (SiLU×mul)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(iters):
    out = LigerSiLUMulFunction.apply(a, b)
torch.cuda.synchronize()
swiglu_ms = (time.time() - t0) * 1000 / iters

print(f"Pure Activation (batch={batch_size}, dim={dim})")
print(f"  ReLU:   {relu_ms:.3f} ms/op")
print(f"  SwiGLU: {swiglu_ms:.3f} ms/op")
