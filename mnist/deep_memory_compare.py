import torch
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_swiglu_layer import FusedSwiGLU

DEVICE = 'cuda'

def build_deep_liger(D, layers):
    """
    Pre-build raw weight tensors for a D→D multi-layer Liger SiLU-GEMM network.
    Returns a list of (Wa, Wb) tuples plus final projection W2.
    """
    Ws = []
    for _ in range(layers):
        Wa = torch.randn(D, D, device=DEVICE, requires_grad=True)
        Wb = torch.randn(D, D, device=DEVICE, requires_grad=True)
        Ws.append((Wa, Wb))
    W2 = torch.randn(D, D, device=DEVICE, requires_grad=True)
    return Ws, W2

def build_deep_fused(D, layers, block=64):
    """
    Pre-build a list of FusedSwiGLU modules plus final projection W2.
    """
    modules = []
    for _ in range(layers):
        m = FusedSwiGLU(in_features=D, hidden_dim=D, block=block).to(DEVICE)
        # Ensure grads are tracked
        for p in m.parameters():
            p.requires_grad_(True)
        modules.append(m)
    W2 = torch.randn(D, D, device=DEVICE, requires_grad=True)
    return modules, W2

def peak_memory(fn):
    """
    Runs fn() once to warm up, then resets stats and reruns fn() to
    return the true max_memory_allocated (in MB).
    """
    # Warm‑up pass (compiles Triton kernels, allocates caches, etc.)
    fn()
    torch.cuda.synchronize()

    # Reset and measure
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()

    return torch.cuda.max_memory_allocated() / (1024**2)

def run_deep_liger(B, D, layers):
    Ws, W2 = build_deep_liger(D, layers)
    def step():
        x = torch.randn(B, D, device=DEVICE, requires_grad=True)
        # forward through each SiLU‑GEMM
        for Wa, Wb in Ws:
            a = x @ Wa
            b = x @ Wb
            x = LigerSiLUMulFunction.apply(a, b)
        # final projection + backward
        y = x @ W2
        y.sum().backward()
    return peak_memory(step)

def run_deep_fused(B, D, layers, block=64):
    modules, W2 = build_deep_fused(D, layers, block)
    def step():
        x = torch.randn(B, D, device=DEVICE, requires_grad=True)
        # forward through each fused SwiGLU
        for m in modules:
            x = m(x)
        # final projection + backward
        y = x @ W2
        y.sum().backward()
    return peak_memory(step)

if __name__ == "__main__":
    batch = 256
    dim   = 2048
    layer_counts = [4, 8, 16, 32, 64, 128, 256]

    # Optionally clear cache between different layer‑counts
    for layers in layer_counts:
        torch.cuda.empty_cache()
        mem_l = run_deep_liger(batch, dim, layers)
        torch.cuda.empty_cache()
        mem_f = run_deep_fused(batch, dim, layers)
        saved = mem_l - mem_f
        print(f"{layers:3d} layers: Liger peak {mem_l:6.1f} MB   Fused peak {mem_f:6.1f} MB   Memory saved {mem_l-mem_f:6.1f} MB")
