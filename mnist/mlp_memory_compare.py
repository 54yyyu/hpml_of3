import torch
import triton
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_swiglu_layer import FusedSwiGLU

def report_peak(fn, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return peak / (1024**2)  # in MB

def run_liger(B, D):
    # two branch weights + output proj
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

def run_fused(B, D, block=64):
    fused = FusedSwiGLU(in_features=D, hidden_dim=D, block=block).cuda()
    W2    = torch.randn(D, D, device='cuda', requires_grad=True)
    x     = torch.randn(B, D, device='cuda', requires_grad=True)

    def step():
        z = fused(x)
        y = z @ W2
        y.sum().backward()
    return report_peak(step)

if __name__ == "__main__":
    for B in [64, 256, 1024]:
        for D in [2048, 4096]:
            mem_l = run_liger(B, D)
            mem_f = run_fused(B, D)
            print(f"B={B:<4} D={D:<4}  Liger peak {mem_l:6.1f} MB   Fused peak {mem_f:6.1f} MB   Difference (Liger vs. Fused) {mem_f-mem_l:6.1f} MB")
