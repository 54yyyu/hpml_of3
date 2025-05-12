import argparse
import torch
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
    parser.add_argument("--output", type=str, default="", help="Output file for memory data (CSV)")
    args = parser.parse_args()

    B = args.batch
    
    # Print header for CSV output
    print("dim,mem_liger_mb,mem_fused_mb,mem_ratio")
    
    # If output file is specified, also write to file
    output_file = None
    if args.output:
        output_file = open(args.output, "w")
        output_file.write("dim,mem_liger_mb,mem_fused_mb,mem_ratio\n")

    for D in args.dims:
        mem_l = run_liger(B, D)
        mem_f = run_fused(B, D, args.block)
        ratio = mem_l / mem_f if mem_f > 0 else float('inf')
        
        # Print to console in CSV format
        print(f"{D},{mem_l:.4f},{mem_f:.4f},{ratio:.4f}")
        
        # Also write to file if specified
        if output_file:
            output_file.write(f"{D},{mem_l:.4f},{mem_f:.4f},{ratio:.4f}\n")
            output_file.flush()  # Ensure data is written immediately

    if output_file:
        output_file.close()
        
    print("\nSummary (more readable format):")
    print(f"{'Dimension':<10} {'Liger (MB)':<12} {'Fused (MB)':<12} {'Ratio':<10}")
    print("-" * 44)
    
    # Run benchmarks again for summary
    for D in args.dims:
        mem_l = run_liger(B, D)
        mem_f = run_fused(B, D, args.block)
        ratio = mem_l / mem_f if mem_f > 0 else float('inf')
        print(f"{D:<10} {mem_l:<12.2f} {mem_f:<12.2f} {ratio:<10.2f}×")

if __name__ == "__main__":
    main()