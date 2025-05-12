import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import wandb
import triton  # for grid calculations
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_swiglu_layer import FusedSwiGLU

def report_peak(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / (1024 ** 2)

def run_liger(B, D):
    W1a = torch.randn(D, D, device='cuda', requires_grad=True)
    W1b = torch.randn(D, D, device='cuda', requires_grad=True)
    W2  = torch.randn(D, D, device='cuda', requires_grad=True)
    x   = torch.randn(B, D, device='cuda', requires_grad=True)
    def step():
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        y = z @ W2
        y.sum().backward()
    return report_peak(step)

def run_fused(B, D, block):
    fused = FusedSwiGLU(in_features=D, hidden_dim=D, block=block).cuda()
    W2    = torch.randn(D, D, device='cuda', requires_grad=True)
    x     = torch.randn(B, D, device='cuda', requires_grad=True)
    def step():
        z = fused(x)
        y = z @ W2
        y.sum().backward()
    return report_peak(step)

def main():
    parser = argparse.ArgumentParser(description="Memory ratio: Liger vs Fused SwiGLU")
    parser.add_argument(
        "--dims", nargs="+", type=int,
        default=[2**i for i in range(9, 15)],  # 512, 1024, 2048, 4096, 8192, 16384
        help="Hidden dimensions (powers of two) to test"
    )
    parser.add_argument("--batch", type=int, default=64, help="Fixed batch size")
    parser.add_argument("--block", type=int, default=64, help="Triton block size")
    args = parser.parse_args()

    B = args.batch
    wandb.init(
        project="activation-microbench",
        config={"batch": B, "dims": args.dims, "block": args.block},
        save_code=True
    )

    # Define metrics with dim as x‐axis
    wandb.define_metric("mem_liger_mb",  step_metric="dim")
    wandb.define_metric("mem_fused_mb",  step_metric="dim")
    wandb.define_metric("mem_ratio_mb",  step_metric="dim")

    for D in args.dims:
        mem_l = run_liger(B, D)
        mem_f = run_fused(B, D, args.block)
        ratio = mem_l / mem_f if mem_f > 0 else float('inf')

        wandb.log({
            "dim":          D,
            "mem_liger_mb": mem_l,
            "mem_fused_mb": mem_f,
            "mem_ratio_mb": ratio
        })

        print(f"B={B} D={D:<6} → ratio (liger/fused): {ratio:.2f}×")

    wandb.finish()

if __name__ == "__main__":
    main()