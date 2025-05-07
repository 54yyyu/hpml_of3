# bench_liger_vs_relu.py

import torch
import time
import triton
import triton.language as tl
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

def bench_pytorch(B, D, iters=200):
    # PyTorch two‐layer MLP with ReLU
    W1 = torch.randn(D, D, device='cuda')
    b1 = torch.randn(D,    device='cuda')
    W2 = torch.randn(D, D, device='cuda')
    x  = torch.randn(B, D, device='cuda')

    # warm-up
    for _ in range(5):
        y = torch.relu(x @ W1 + b1)
        _ = y @ W2
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    for _ in range(iters):
        y = torch.relu(x @ W1 + b1)
        out = y @ W2
        torch.cuda.synchronize()
        _ = out.sum().item()
    print(f"PyTorch ReLU       B={B:<3d} D={D:<4d} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


def bench_liger(B, D, iters=200):
    # two branch linears + LigerSiLUMulFunction + final linear
    W1a = torch.randn(D, D, device='cuda')
    W1b = torch.randn(D, D, device='cuda')
    W2  = torch.randn(D, D, device='cuda')
    x   = torch.randn(B, D, device='cuda')

    # warm-up
    for _ in range(5):
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        _ = z @ W2
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    for _ in range(iters):
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        out = z @ W2
        torch.cuda.synchronize()
        _ = out.sum().item()
    print(f"Liger-SwiGLU       B={B:<3d} D={D:<4d} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


if __name__ == "__main__":
    print("\nBenchmark LigerSiLUMulFunction vs ReLU on your L4:\n")
    for B, D in [(64, 4096), (128, 4096), (256, 4096)]:
        bench_pytorch(B, D)
        bench_liger(B, D)
