import torch
import triton
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_swiglu_layer import FusedSwiGLU

DEVICE = 'cuda'

def peak_memory(fn):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)

def run_deep_liger(B, D, layers):
    def step():
        x = torch.randn(B, D, device=DEVICE, requires_grad=True)
        W2 = torch.randn(D, D, device=DEVICE, requires_grad=True)
        # build random weights for each layer
        Ws = []
        for _ in range(layers):
            Wa = torch.randn(D, D, device=DEVICE, requires_grad=True)
            Wb = torch.randn(D, D, device=DEVICE, requires_grad=True)
            Ws.append((Wa, Wb))
        # forward
        for Wa, Wb in Ws:
            a = x @ Wa
            b = x @ Wb
            x = LigerSiLUMulFunction.apply(a, b)
        y = x @ W2
        y.sum().backward()
    return peak_memory(step)

def run_deep_fused(B, D, layers, block=64):
    def step():
        x = torch.randn(B, D, device=DEVICE, requires_grad=True)
        W2 = torch.randn(D, D, device=DEVICE, requires_grad=True)
        # build fused layers
        fused_layers = [
            FusedSwiGLU(in_features=D, hidden_dim=D, block=block).to(DEVICE)
            for _ in range(layers)
        ]
        # forward
        for layer in fused_layers:
            x = layer(x)
        y = x @ W2
        y.sum().backward()
    return peak_memory(step)

if __name__ == "__main__":
    batch = 256
    dim   = 2048
    for layers in [4, 8, 16]:
        mem_l = run_deep_liger(batch, dim, layers)
        mem_f = run_deep_fused(batch, dim, layers)
        print(f"{layers:2d} layers: Liger peak {mem_l:6.1f} MB   Fused peak {mem_f:6.1f} MB   Memory saved {mem_l-mem_f:6.1f} MB")
