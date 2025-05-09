import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, time, triton
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_swiglu_layer import FusedSwiGLU

def bench_liger(B, D, iters=100):
    # two branch weights + output projection
    W1a = nn_param = torch.randn(D, D, device='cuda', requires_grad=True)
    W1b = torch.randn(D, D, device='cuda', requires_grad=True)
    W2  = torch.randn(D, D, device='cuda', requires_grad=True)
    x   = torch.randn(B, D, device='cuda', requires_grad=True)

    # warm‐up
    for _ in range(10):
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        y = z @ W2
        y.sum().backward()
        # zero grads
        x.grad = W1a.grad = W1b.grad = W2.grad = None
    torch.cuda.synchronize()

    # timed loop
    t0 = time.perf_counter()
    for _ in range(iters):
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        y = z @ W2
        y.sum().backward()
        # zero grads
        x.grad = W1a.grad = W1b.grad = W2.grad = None
    torch.cuda.synchronize()
    print(f"Liger-bwd  B={B:<4} D={D:<4} -> {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


def bench_fused(B, D, iters=100, block=64):
    # fused SwiGLU layer + same output projection
    fused = FusedSwiGLU(in_features=D, hidden_dim=D, block=block).cuda()
    W2    = torch.randn(D, D, device='cuda', requires_grad=True)
    x     = torch.randn(B, D, device='cuda', requires_grad=True)

    # warm‐up
    for _ in range(10):
        z = fused(x)
        y = z @ W2
        y.sum().backward()
        x.grad = W2.grad = None
        fused.Wf.grad = None
    torch.cuda.synchronize()

    # timed loop
    t0 = time.perf_counter()
    for _ in range(iters):
        z = fused(x)
        y = z @ W2
        y.sum().backward()
        x.grad = W2.grad = None
        fused.Wf.grad = None
    torch.cuda.synchronize()
    print(f"Fused-bwd B={B:<4} D={D:<4} -> {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


if __name__ == "__main__":
    for B in [64, 256, 1024]:
        for D in [2048, 4096]:
            bench_liger(B, D)
            bench_fused(B, D)
