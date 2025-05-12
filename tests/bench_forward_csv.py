import argparse
import time
import torch
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
    parser.add_argument("--output",  type=str, default="",  help="Output file for timing data (CSV)")
    args = parser.parse_args()

    B = 64  # fixed batch size
    
    # Print header for CSV output
    print("dim,liger_ms,fused_ms,speedup")
    
    # If output file is specified, also write to file
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        output_file.write("dim,liger_ms,fused_ms,speedup\n")

    for D in args.dims:
        liger_ms = bench_liger(B, D, args.iters, args.warmups)
        fused_ms = bench_fused(B, D, args.iters, args.warmups, args.block)
        speedup  = liger_ms / fused_ms

        # Print to console in CSV format
        print(f"{D},{liger_ms:.4f},{fused_ms:.4f},{speedup:.4f}")
        
        # Also write to file if specified
        if output_file:
            output_file.write(f"{D},{liger_ms:.4f},{fused_ms:.4f},{speedup:.4f}\n")
            output_file.flush()  # Ensure data is written immediately

    if output_file:
        output_file.close()
        
    print("\nSummary (more readable format):")
    print(f"{'Dimension':<10} {'Liger (ms)':<12} {'Fused (ms)':<12} {'Speedup':<10}")
    print("-" * 44)
    
    # Run benchmarks again for summary
    for D in args.dims:
        liger_ms = bench_liger(B, D, args.iters, args.warmups)
        fused_ms = bench_fused(B, D, args.iters, args.warmups, args.block)
        speedup  = liger_ms / fused_ms
        print(f"{D:<10} {liger_ms:<12.4f} {fused_ms:<12.4f} {speedup:<10.4f}×")

if __name__ == "__main__":
    main()