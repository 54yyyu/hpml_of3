import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, time, triton
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_forward_kernel import swiglu_fused_forward_kernel

def bench_liger(B, D, iters=100):
    W1a, W1b, W2 = [torch.randn(D, D, device='cuda') for _ in range(3)]
    x = torch.randn(B, D, device='cuda')
    # warm
    for _ in range(10):
        z = LigerSiLUMulFunction.apply(x @ W1a, x @ W1b)
        _ = z @ W2
    torch.cuda.synchronize()
    # time
    t0 = time.perf_counter()
    for _ in range(iters):
        z = LigerSiLUMulFunction.apply(x @ W1a, x @ W1b)
        _ = z @ W2
    torch.cuda.synchronize()
    print(f"Liger  B={B:<4} D={D:<4} -> {(time.perf_counter()-t0)/iters*1000:6.2f} ms")

def bench_fused(B, D, iters=100, block=64):
    Wf = torch.randn(D, 2*D, device='cuda')
    x  = torch.randn(B, D, device='cuda')
    Z  = torch.empty(B, D, device='cuda')
    grid = (triton.cdiv(B,block), triton.cdiv(D,block))
    # warm
    for _ in range(10):
        swiglu_fused_forward_kernel[grid](x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block)
        _ = Z @ Wf[:, :D]
    torch.cuda.synchronize()
    # time
    t0 = time.perf_counter()
    for _ in range(iters):
        swiglu_fused_forward_kernel[grid](x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block)
        _ = Z @ Wf[:, :D]
    torch.cuda.synchronize()
    print(f"Fused  B={B:<4} D={D:<4} -> {(time.perf_counter()-t0)/iters*1000:6.2f} ms")

if __name__ == "__main__":
    for B in [64, 256, 1024]:
        for D in [2048, 4096]:
            bench_liger(B, D)
            bench_fused(B, D)
