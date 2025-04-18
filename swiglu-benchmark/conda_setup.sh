#!/bin/bash

# Create the conda environment
conda create -n swiglu-benchmark python=3.10 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate swiglu-benchmark

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
pip install triton

# Install JAX with CUDA support
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install numpy matplotlib pandas seaborn tqdm

# Create directories if they don't exist
mkdir -p implementations results

echo "Environment setup complete. Activate with: conda activate swiglu-benchmark"