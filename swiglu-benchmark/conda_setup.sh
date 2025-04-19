#!/bin/bash

# Load modules
module load cuda12.2/
module load mamba

# Initialize mamba
mamba init
source /burg/home/yy3448/.bashrc

# Create the mamba environment
if mamba env list | grep -q 'hpml'; then
    echo "Environment 'hpml' found. Activating it..."
    mamba activate hpml
else
    echo "Environment 'hpml' not found. Creating it..."
    mamba create -n hpml python=3.11 -y
    mamba activate hpml
fi

# Install PyTorch with CUDA
mamba install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia --yes

# Install Triton
pip install triton

# Install JAX with CUDA support
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax

# Install other dependencies
pip install numpy matplotlib pandas seaborn tqdm

echo "Environment setup complete. Activate with: mamba activate hpml"