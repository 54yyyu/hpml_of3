#!/bin/bash

echo "Running main.py... (RELU)"
echo ""
python3 main.py

echo "Running liger_swiglu_main.py... (LIGER SWIGLU)"
echo ""
python3 liger_swiglu_main.py

echo "Running fused_swiglu_main.py... (FUSED SWIGLU w/o weight-grad)"
echo ""
python3 fused_swiglu_main.py

echo "Running fused_swiglu_main.py... (FUSED SWIGLU w weight-grad)"
echo ""
python3 fused_swiglu_main_WF.py

echo "mlp_forward_compare.py"
python mlp_forward_compare.py 

echo "mlp_backward_compare.py"
python mlp_backward_compare.py 

echo "mlp_backward_compare_WF.py"
python mlp_backward_compare_WF.py 

echo ""
echo "All scripts completed!"