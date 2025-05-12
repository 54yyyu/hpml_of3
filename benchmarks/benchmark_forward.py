import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import torch
import wandb
import triton
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_forward_kernel import swiglu_fused_forward_kernel

def bench_liger(B, D, iters=100, warmups=10):
    W1a = torch.randn(D, D, device='cuda')
    W1b = torch.randn(D, D, device='cuda')
    W2  = torch.randn(D, D, device='cuda')
    x   = torch.randn(B, D, device='cuda')
    for _ in range(warmups):
        z = LigerSiLUMulFunction.apply(x @ W1a, x @ W1b)
        _ = z @ W2
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        z = LigerSiLUMulFunction.apply(x @ W1a, x @ W1b)
        _ = z @ W2
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

def bench_fused(B, D, iters=100, warmups=10, block=256):
    Wf = torch.randn(D, 2*D, device='cuda')
    x  = torch.randn(B, D, device='cuda')
    Z  = torch.empty(B, D, device='cuda')
    grid = (triton.cdiv(B, block), triton.cdiv(D, block))
    for _ in range(warmups):
        swiglu_fused_forward_kernel[grid](
            x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0),  Z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        _ = Z @ Wf[:, :D]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        swiglu_fused_forward_kernel[grid](
            x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0),  Z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        _ = Z @ Wf[:, :D]
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0

def main():
    parser = argparse.ArgumentParser(description="Compare Liger vs Fused SwiGLU Performance")
    parser.add_argument(
        "--dims", nargs="+", type=int,
        default=[2**i for i in range(9, 16)],  # 512..32768
        help="Hidden dimensions (powers of two) to test"
    )
    parser.add_argument("--iters",   type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmups", type=int, default=10,  help="Warm-up iterations")
    parser.add_argument("--block",   type=int, default=64,  help="Triton block size for fused kernel")
    args = parser.parse_args()

    B = 64  # fixed batch size
    wandb.init(
        project="swiglu-kernel-compare",
        config={
            "batch": B,
            "dims": args.dims,
            "iters": args.iters,
            "warmups": args.warmups,
            "block": args.block,
        },
        save_code=True
    )

    # Ensure 'dim' is used as the x‐axis for speedup
    wandb.define_metric("speedup", step_metric="dim")

    for D in args.dims:
        liger_ms = bench_liger(B, D, args.iters, args.warmups)
        fused_ms = bench_fused(B, D, args.iters, args.warmups, args.block)
        speedup  = liger_ms / fused_ms

        # log with 'dim' so it becomes the x-axis
        wandb.log({
            "dim":      D,
            "speedup":  speedup
        })

        print(f"B={B} D={D:<6} → speedup (liger/fused): {speedup:.2f}×")

    wandb.finish()

if __name__ == "__main__":
    main()