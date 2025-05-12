#!/bin/bash

# Set up error handling
echo "Starting the tests..."

# MLP comparisons
echo "Running MLP forward comparison..."
python mlp_forward_compare.py

echo "Running MLP backward comparison..."
python mlp_backward_compare.py

echo "Running MLP backward comparison with weight freezing..."
python mlp_backward_compare_WF.py

# SwiGLU tests
echo "Running Liger SwiGLU test (first run)..."
python liger_swiglu_main.py

echo "Running Fused SwiGLU test..."
python fused_swiglu_main.py

echo "Running Liger SwiGLU test (second run)..."
python liger_swiglu_main.py

# Memory comparisons
echo "Running MLP memory comparison..."
python mlp_memory_compare.py

echo "Running deep memory comparison..."
python deep_memory_compare.py

# Numerical test
echo "Running small numerical test..."
python small_numerical_test.py

echo "All tests completed successfully!"