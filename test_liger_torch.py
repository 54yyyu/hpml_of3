import torch
import time
from torch import nn
from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

# 1. Synthetic input
batch, dim = 32, 4096  # adjust as needed
a = torch.randn(batch, dim, device='cuda', requires_grad=True)
b = torch.randn(batch, dim, device='cuda', requires_grad=True)
grad_out = torch.randn_like(a)

# 2. Warm‑up
for _ in range(5):
    c = nn.functional.silu(a) * b
    c.mean().backward()
    torch.cuda.synchronize()
    a.grad.zero_(); b.grad.zero_()

# 3. Time standard PyTorch
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
c = nn.functional.silu(a) * b
loss = c.mean(); loss.backward()
end.record(); torch.cuda.synchronize()
print("PyTorch SwiGLU: ",
      start.elapsed_time(end), "ms")

# 4. Zero grads
a.grad.zero_(); b.grad.zero_()

# 5. Warm‑up Liger
for _ in range(5):
    aa, bb, cc = swiglu_forward(a, b)
    grad = grad_out.clone()
    aa, bb = swiglu_backward(a, b, grad)
    torch.cuda.synchronize()

# 6. Time Liger SwiGLU
start.record()
aa, bb, cc = swiglu_forward(a, b)
aa, bb = swiglu_backward(a, b, grad_out)
end.record(); torch.cuda.synchronize()
print("Liger SwiGLU:   ",
      start.elapsed_time(end), "ms")
