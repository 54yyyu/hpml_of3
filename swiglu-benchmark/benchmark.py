"""
SwiGLU Benchmark

This script benchmarks different SwiGLU implementations for both
speed and memory usage across various input sizes.
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer
import gc
import json
from datetime import datetime

# Import implementations
from implementations.pytorch_swiglu import (
    PyTorchSwiGLU,
    MemoryOptimizedSwiGLU as PyTorchMemoryOptimizedSwiGLU,
    CheckpointedSwiGLU as PyTorchCheckpointedSwiGLU
)
from implementations.liger_swiglu import LigerSwiGLU
from implementations.unsloth_swiglu import UnslothSwiGLU

# Only import JAX implementations if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from implementations.alphafold3_swiglu import (
        AlphaFold3SwiGLU,
        MemoryOptimizedSwiGLU as JAXMemoryOptimizedSwiGLU,
        gated_linear_unit,
        PallasGatedLinearUnit,
        GatedLinearUnit
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not found, skipping AlphaFold3 implementations")


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if HAS_JAX:
        import random
        random.seed(seed)
        jax.random.PRNGKey(seed)


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, name, input_size, batch_size, seq_len, hidden_dim, output_dim,
                  forward_time, backward_time, peak_memory, model_type):
        """Add a benchmark result."""
        self.results.append({
            'name': name,
            'input_size': input_size,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': forward_time + backward_time,
            'peak_memory': peak_memory,
            'model_type': model_type
        })
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def save_to_csv(self, filename):
        """Save results to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
    
    def save_to_json(self, filename):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def plot_results(self, output_dir):
        """Plot and save benchmark results."""
        df = self.to_dataframe()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 14})
        
        # Forward time comparison by model
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='name', y='forward_time', hue='input_size', data=df)
        chart.set_title('Forward Pass Time by Implementation')
        chart.set_xlabel('Implementation')
        chart.set_ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forward_time.png'), dpi=300)
        plt.close()
        
        # Backward time comparison
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='name', y='backward_time', hue='input_size', data=df)
        chart.set_title('Backward Pass Time by Implementation')
        chart.set_xlabel('Implementation')
        chart.set_ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'backward_time.png'), dpi=300)
        plt.close()
        
        # Total time comparison
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='name', y='total_time', hue='input_size', data=df)
        chart.set_title('Total Time (Forward + Backward) by Implementation')
        chart.set_xlabel('Implementation')
        chart.set_ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_time.png'), dpi=300)
        plt.close()
        
        # Memory usage comparison
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='name', y='peak_memory', hue='input_size', data=df)
        chart.set_title('Peak Memory Usage by Implementation')
        chart.set_xlabel('Implementation')
        chart.set_ylabel('Memory (MB)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'peak_memory.png'), dpi=300)
        plt.close()
        
        # Memory vs Time scatter plot
        plt.figure(figsize=(12, 8))
        chart = sns.scatterplot(x='total_time', y='peak_memory', hue='name', 
                               style='input_size', s=100, data=df)
        chart.set_title('Memory Usage vs. Computation Time')
        chart.set_xlabel('Total Time (ms)')
        chart.set_ylabel('Peak Memory (MB)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_vs_time.png'), dpi=300)
        plt.close()
        
        # Model type comparison
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x='model_type', y='total_time', data=df)
        chart.set_title('Performance by Model Type')
        chart.set_xlabel('Model Type')
        chart.set_ylabel('Total Time (ms)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_type_comparison.png'), dpi=300)
        plt.close()


def benchmark_pytorch_implementation(model_class, model_name, model_type, 
                                    batch_size, seq_len, hidden_dim, output_dim, 
                                    num_warmup, num_repeats, results):
    """Benchmark a PyTorch SwiGLU implementation."""
    print(f"Benchmarking {model_name}...")
    
    # Create model and input
    model = model_class(hidden_dim, output_dim).cuda()
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    
    # Warmup
    for _ in range(num_warmup):
        y = model(x)
        loss = y.sum()
        loss.backward()
    
    clear_gpu_memory()
    
    # Measure forward pass time
    torch.cuda.synchronize()
    start_memory = get_gpu_memory_usage()
    start_time = time.time()
    
    forward_times = []
    backward_times = []
    peak_memories = []
    
    for _ in range(num_repeats):
        # Forward pass
        torch.cuda.synchronize()
        t0 = time.time()
        y = model(x)
        torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append((t1 - t0) * 1000)  # Convert to ms
        
        # Backward pass
        loss = y.sum()
        torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        backward_times.append((t1 - t0) * 1000)  # Convert to ms
        
        # Memory usage
        peak_memories.append(get_gpu_memory_usage() - start_memory)
        
        # Clear gradients for next iteration
        model.zero_grad()
    
    # Calculate averages
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    avg_peak_memory = sum(peak_memories) / len(peak_memories)
    
    # Add result
    input_size = f"{batch_size}x{seq_len}x{hidden_dim}"
    results.add_result(
        name=model_name,
        input_size=input_size,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        forward_time=avg_forward_time,
        backward_time=avg_backward_time,
        peak_memory=avg_peak_memory,
        model_type=model_type
    )
    
    clear_gpu_memory()
    del model, x, y
    return avg_forward_time, avg_backward_time, avg_peak_memory


def benchmark_jax_implementation(model_class, model_name, model_type,
                                batch_size, seq_len, hidden_dim, output_dim,
                                num_warmup, num_repeats, results):
    """Benchmark a JAX SwiGLU implementation."""
    if not HAS_JAX:
        print(f"Skipping {model_name} (JAX not available)")
        return
    
    print(f"Benchmarking {model_name}...")
    
    # Initialize model
    key = jax.random.PRNGKey(0)
    model = model_class(features=output_dim)
    x = jax.random.normal(key, (batch_size, seq_len, hidden_dim))
    
    # Initialize parameters
    variables = model.init(key, x)
    
    # Define forward and loss function
    def forward_fn(params, x):
        return model.apply(params, x)
    
    def loss_fn(params, x):
        y = forward_fn(params, x)
        return jnp.sum(y)
    
    # Create JIT-compiled versions
    forward_jit = jax.jit(forward_fn)
    backward_jit = jax.jit(jax.value_and_grad(loss_fn))
    
    # Warmup
    for _ in range(num_warmup):
        forward_jit(variables, x).block_until_ready()
        backward_jit(variables, x)[0].block_until_ready()
    
    # Measure performance
    forward_times = []
    backward_times = []
    
    # JAX doesn't provide direct memory tracking like PyTorch,
    # so we'll use a fixed value based on device inspection
    # This is a limitation of the benchmark
    peak_memory = 0
    
    for _ in range(num_repeats):
        # Forward pass
        start_time = time.time()
        y = forward_jit(variables, x).block_until_ready()
        end_time = time.time()
        forward_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Backward pass
        start_time = time.time()
        _, grad = backward_jit(variables, x)
        grad = jax.tree_map(lambda x: x.block_until_ready(), grad)
        end_time = time.time()
        backward_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate averages
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    
    # Add result
    input_size = f"{batch_size}x{seq_len}x{hidden_dim}"
    results.add_result(
        name=model_name,
        input_size=input_size,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        forward_time=avg_forward_time,
        backward_time=avg_backward_time,
        peak_memory=0,  # Unable to measure directly in JAX
        model_type=model_type
    )


def benchmark_jax_gated_linear_unit(model_name, implementation, model_type,
                                   batch_size, seq_len, hidden_dim, output_dim,
                                   num_warmup, num_repeats, results):
    """Benchmark JAX GatedLinearUnit implementations directly.
    
    This allows testing the core gated_linear_unit function with different
    implementation options.
    
    Args:
        model_name: Name of the implementation for reporting
        implementation: Implementation to use ('xla' or 'triton')
        model_type: Type of model for categorization
        batch_size, seq_len, hidden_dim, output_dim: Dimensions for the input
        num_warmup, num_repeats: Number of warm-up and benchmark iterations
        results: BenchmarkResult object to store results
    """
    if not HAS_JAX:
        print(f"Skipping {model_name} (JAX not available)")
        return
    
    print(f"Benchmarking {model_name}...")
    
    # Initialize input and weights
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))
    weight = jax.random.normal(key2, (hidden_dim, 2, output_dim)) * 0.02
    
    # Define functions for benchmarking
    def forward_fn(x, weight):
        return gated_linear_unit(
            x=x,
            weight=weight,
            activation=jax.nn.swish,
            precision=None,
            implementation=implementation
        )
    
    def loss_fn(x, weight):
        y = forward_fn(x, weight)
        return jnp.sum(y)
    
    # Create JIT-compiled versions
    forward_jit = jax.jit(forward_fn)
    backward_jit = jax.jit(jax.value_and_grad(loss_fn, argnums=1))
    
    # Warmup
    for _ in range(num_warmup):
        forward_jit(x, weight).block_until_ready()
        backward_jit(x, weight)[0].block_until_ready()
    
    # Measure performance
    forward_times = []
    backward_times = []
    
    for _ in range(num_repeats):
        # Forward pass
        start_time = time.time()
        y = forward_jit(x, weight).block_until_ready()
        end_time = time.time()
        forward_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Backward pass
        start_time = time.time()
        _, grad = backward_jit(x, weight)
        grad = jax.tree_map(lambda x: x.block_until_ready(), grad)
        end_time = time.time()
        backward_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate averages
    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)
    
    # Add result
    input_size = f"{batch_size}x{seq_len}x{hidden_dim}"
    results.add_result(
        name=model_name,
        input_size=input_size,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        forward_time=avg_forward_time,
        backward_time=avg_backward_time,
        peak_memory=0,  # Unable to measure directly in JAX
        model_type=model_type
    )


def run_benchmarks(configs, num_warmup=5, num_repeats=10):
    """Run benchmarks for all implementations and configurations."""
    results = BenchmarkResult()
    
    for config in configs:
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        hidden_dim = config['hidden_dim']
        output_dim = config['output_dim']
        
        print(f"\nBenchmarking configuration: batch_size={batch_size}, "
              f"seq_len={seq_len}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        
        # PyTorch implementations
        benchmark_pytorch_implementation(
            PyTorchSwiGLU, "PyTorch-Standard", "PyTorch",
            batch_size, seq_len, hidden_dim, output_dim,
            num_warmup, num_repeats, results
        )
        
        benchmark_pytorch_implementation(
            PyTorchMemoryOptimizedSwiGLU, "PyTorch-MemoryOpt", "Memory-Optimized",
            batch_size, seq_len, hidden_dim, output_dim,
            num_warmup, num_repeats, results
        )
        
        benchmark_pytorch_implementation(
            PyTorchCheckpointedSwiGLU, "PyTorch-Checkpointed", "Checkpointed",
            batch_size, seq_len, hidden_dim, output_dim,
            num_warmup, num_repeats, results
        )
        
        # Triton implementations - wrapped in try/except as they require GPU hardware
        try:
            benchmark_pytorch_implementation(
                LigerSwiGLU, "Liger-Triton", "Triton",
                batch_size, seq_len, hidden_dim, output_dim,
                num_warmup, num_repeats, results
            )
        except Exception as e:
            print(f"Skipping Liger-Triton (Error: {e})")
        
        try:
            benchmark_pytorch_implementation(
                UnslothSwiGLU, "Unsloth-Triton", "Triton",
                batch_size, seq_len, hidden_dim, output_dim,
                num_warmup, num_repeats, results
            )
        except Exception as e:
            print(f"Skipping Unsloth-Triton (Error: {e})")
        
        # JAX implementations
        if HAS_JAX:
            benchmark_jax_implementation(
                AlphaFold3SwiGLU, "AlphaFold3-Standard", "JAX",
                batch_size, seq_len, hidden_dim, output_dim,
                num_warmup, num_repeats, results
            )
            
            benchmark_jax_implementation(
                JAXMemoryOptimizedSwiGLU, "AlphaFold3-MemoryOpt", "JAX-Checkpointed",
                batch_size, seq_len, hidden_dim, output_dim,
                num_warmup, num_repeats, results
            )
            
            # Test direct gated_linear_unit implementations
            benchmark_jax_gated_linear_unit(
                "AlphaFold3-GLU-XLA", "xla", "JAX-Direct",
                batch_size, seq_len, hidden_dim, output_dim,
                num_warmup, num_repeats, results
            )
            
            # Only run Triton implementation on GPU platforms
            if jax.devices()[0].platform == 'gpu':
                try:
                    benchmark_jax_gated_linear_unit(
                        "AlphaFold3-GLU-Triton", "triton", "JAX-Triton",
                        batch_size, seq_len, hidden_dim, output_dim,
                        num_warmup, num_repeats, results
                    )
                except Exception as e:
                    print(f"Skipping AlphaFold3-GLU-Triton (Error: {e})")
    
    return results


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description='SwiGLU Benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. Please run on a GPU for meaningful benchmarks.")
        return
    
    # Define benchmark configurations
    small_config = {
        'batch_size': 32,
        'seq_len': 128,
        'hidden_dim': 512,
        'output_dim': 512
    }
    
    medium_config = {
        'batch_size': 16,
        'seq_len': 512,
        'hidden_dim': 1024,
        'output_dim': 1024
    }
    
    large_config = {
        'batch_size': 8,
        'seq_len': 1024,
        'hidden_dim': 2048,
        'output_dim': 2048
    }
    
    # AlphaFold3-like config
    alphafold3_config = {
        'batch_size': 4,
        'seq_len': 256,
        'hidden_dim': 4096,
        'output_dim': 4096
    }
    
    configs = [small_config, medium_config, large_config, alphafold3_config]
    
    # Run benchmarks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_benchmarks(configs)
    
    # Save results
    results.save_to_csv(os.path.join(output_dir, 'results.csv'))
    results.save_to_json(os.path.join(output_dir, 'results.json'))
    results.plot_results(output_dir)
    
    print(f"\nBenchmark completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()