# SwiGLU Benchmark Suite

This repository contains implementations and benchmarks for various SwiGLU (Swish-Gated Linear Unit) variants used in contemporary AI models.

## Overview

SwiGLU is a variant of the Gated Linear Unit activation function that uses the Swish activation. It is used in many state-of-the-art models including AlphaFold3, Mistral, Llama, and others.

This benchmark suite compares different implementation approaches:
- Standard PyTorch implementation
- Memory-optimized implementation
- Checkpointed implementation
- GPU-optimized implementations (Triton/Pallas)

## Implementations

### AlphaFold3 SwiGLU

The `alphafold3_swiglu.py` file contains a comprehensive implementation of SwiGLU derived from the AlphaFold3 codebase. It features:

1. **Standard Flax module implementation** - A basic implementation using Flax/JAX
2. **Memory-optimized implementation** - Reduces memory usage by combining linear projections
3. **Activation checkpointing** - Trades computation for memory by recomputing activations during backpropagation
4. **XLA and Triton implementations** - Different backend optimizations
5. **Functional API** - Standalone functions for direct use in JAX transformations
6. **GPU-specific optimizations** - The Triton/Pallas implementation includes hardware-aware tuning

Key features from the AlphaFold3 implementation:
- Precision control for matrix multiplication
- Type checking and validation
- Automatic fallback from Triton to XLA when needed
- Support for custom activation functions

### Other Implementations

For comparison, this benchmark also includes:

1. **PyTorch SwiGLU** - Reference implementation using PyTorch
2. **Liger SwiGLU** - Using Triton kernels for GPU optimization
3. **Unsloth SwiGLU** - Alternative Triton-based implementation

## Running Benchmarks

The `benchmark.py` script tests all implementations across various input sizes:

```bash
python benchmark.py --output-dir results
```

The benchmark includes:
- Forward and backward pass timing
- Peak memory usage measurement (for PyTorch implementations)
- Comparison across different input dimensions

## Results

Benchmark results are saved as:
- CSV files with raw measurements
- JSON files with detailed configuration
- Visualizations for easy comparison:
  - Forward/backward pass times
  - Memory usage
  - Time vs memory trade-offs
  - Implementation type comparisons

## Setup

```bash
# Set up the environment
conda env create -f environment.yml
conda activate swiglu-bench

# Run benchmarks
python benchmark.py
```

## License

This benchmark code is licensed under MIT. The AlphaFold3 implementation is derived from DeepMind's AlphaFold3 codebase, which is licensed under CC BY-NC-SA 4.0.
