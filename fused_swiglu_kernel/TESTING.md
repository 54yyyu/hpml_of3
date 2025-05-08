# TESTING

## Output of the tests
All of the tests are done on an NVIDIA L4. The tests are included in test/.
To use it, place them in the same directory as the kernel.

### Direct comparison of forward vs. backward for Liger vs Fused
```
# Forward
python mlp_forward_compare.py 
Liger  B=64   D=2048 →   0.21 ms
Fused  B=64   D=2048 ->   0.07 ms
Liger  B=64   D=4096 →   0.99 ms
Fused  B=64   D=4096 ->   0.35 ms
Liger  B=256  D=2048 →   0.74 ms
Fused  B=256  D=2048 ->   0.27 ms
Liger  B=256  D=4096 →   2.34 ms
Fused  B=256  D=4096 ->   0.83 ms
Liger  B=1024 D=2048 →   3.10 ms
Fused  B=1024 D=2048 ->   1.14 ms
Liger  B=1024 D=4096 →   8.71 ms
Fused  B=1024 D=4096 ->   3.12 ms

# Backward no weight-grad
python mlp_backward_compare.py 
Liger-bwd  B=64   D=2048 ->   0.81 ms
Fused-bwd B=64   D=2048 ->   0.60 ms
Liger-bwd  B=64   D=4096 ->   3.36 ms
Fused-bwd B=64   D=4096 ->   1.01 ms
Liger-bwd  B=256  D=2048 ->   2.21 ms
Fused-bwd B=256  D=2048 ->   0.62 ms
Liger-bwd  B=256  D=4096 ->   8.65 ms
Fused-bwd B=256  D=4096 ->   2.61 ms
Liger-bwd  B=1024 D=2048 ->   8.84 ms
Fused-bwd B=1024 D=2048 ->   2.72 ms
Liger-bwd  B=1024 D=4096 ->  28.86 ms
Fused-bwd B=1024 D=4096 ->   9.22 ms

# Backward with weight-grad
python mlp_backward_compare_WF.py 
Liger-bwd  B=64   D=2048 ->   0.83 ms
Fused-bwd B=64   D=2048 ->   0.66 ms
Liger-bwd  B=64   D=4096 ->   3.38 ms
Fused-bwd B=64   D=4096 ->   1.66 ms
Liger-bwd  B=256  D=2048 ->   2.21 ms
Fused-bwd B=256  D=2048 ->   0.99 ms
Liger-bwd  B=256  D=4096 ->   8.61 ms
Fused-bwd B=256  D=4096 ->   3.87 ms
Liger-bwd  B=1024 D=2048 ->   8.87 ms
Fused-bwd B=1024 D=2048 ->   3.91 ms
Liger-bwd  B=1024 D=4096 ->  28.76 ms
Fused-bwd B=1024 D=4096 ->  13.51 ms
```

### MNIST test output
#### Liger
```
python liger_swiglu_main.py
Epoch 1 train time: 39542.0 ms

Test set: Average loss: 0.0512, Accuracy: 9842/10000 (98%)

Epoch 1 test time:  2050.4 ms

---------------------------------------------

Epoch 2 train time: 37844.2 ms

Test set: Average loss: 0.0351, Accuracy: 9896/10000 (99%)

Epoch 2 test time:  2056.6 ms

---------------------------------------------

Epoch 3 train time: 37840.8 ms

Test set: Average loss: 0.0295, Accuracy: 9907/10000 (99%)

Epoch 3 test time:  2058.4 ms

---------------------------------------------

Epoch 4 train time: 37836.2 ms

Test set: Average loss: 0.0284, Accuracy: 9922/10000 (99%)

Epoch 4 test time:  2092.7 ms

---------------------------------------------
```

#### Fused no weight-grad
```
python fused_swiglu_main.py 
Epoch 1 train time: 15472.0 ms

Test set: Average loss: 0.0700, Accuracy: 9763/10000 (98%)

Epoch 1 test time:  2060.7 ms

---------------------------------------------

Epoch 2 train time: 14404.7 ms

Test set: Average loss: 0.0444, Accuracy: 9853/10000 (99%)

Epoch 2 test time:  2060.2 ms

---------------------------------------------

Epoch 3 train time: 14541.9 ms

Test set: Average loss: 0.0365, Accuracy: 9878/10000 (99%)

Epoch 3 test time:  2087.5 ms

---------------------------------------------

Epoch 4 train time: 14055.5 ms

Test set: Average loss: 0.0350, Accuracy: 9874/10000 (99%)

Epoch 4 test time:  2022.5 ms

---------------------------------------------
```

#### Fused with weight-grad
```
python liger_swiglu_main.py 
Epoch 1 train time: 39542.0 ms

Test set: Average loss: 0.0512, Accuracy: 9842/10000 (98%)

Epoch 1 test time:  2050.4 ms

---------------------------------------------

Epoch 2 train time: 37844.2 ms

Test set: Average loss: 0.0351, Accuracy: 9896/10000 (99%)

Epoch 2 test time:  2056.6 ms

---------------------------------------------

Epoch 3 train time: 37840.8 ms

Test set: Average loss: 0.0295, Accuracy: 9907/10000 (99%)

Epoch 3 test time:  2058.4 ms

---------------------------------------------

Epoch 4 train time: 37836.2 ms

Test set: Average loss: 0.0284, Accuracy: 9922/10000 (99%)

Epoch 4 test time:  2092.7 ms

---------------------------------------------
```

### Memory test output
```
python mlp_memory_compare.py 
B=64   D=2048  Liger peak  115.8 MB   Fused peak   82.8 MB   Difference (Liger vs. Fused)  -33.0 MB
B=64   D=4096  Liger peak  407.3 MB   Fused peak  277.3 MB   Difference (Liger vs. Fused) -130.0 MB
B=256  D=2048  Liger peak  126.3 MB   Fused peak   90.3 MB   Difference (Liger vs. Fused)  -36.0 MB
B=256  D=4096  Liger peak  428.3 MB   Fused peak  292.3 MB   Difference (Liger vs. Fused) -136.0 MB
B=1024 D=2048  Liger peak  168.3 MB   Fused peak  120.3 MB   Difference (Liger vs. Fused)  -48.0 MB
B=1024 D=4096  Liger peak  512.3 MB   Fused peak  352.3 MB   Difference (Liger vs. Fused) -160.0 MB

python deep_memory_compare.py 
  4 layers: Liger peak  352.3 MB   Fused peak  208.3 MB   Memory saved  144.0 MB
  8 layers: Liger peak  632.3 MB   Fused peak  344.3 MB   Memory saved  288.0 MB
 16 layers: Liger peak 1192.3 MB   Fused peak  616.3 MB   Memory saved  576.0 MB
 32 layers: Liger peak 2312.3 MB   Fused peak 1160.3 MB   Memory saved 1152.0 MB
 64 layers: Liger peak 4552.3 MB   Fused peak 2248.3 MB   Memory saved 2304.0 MB
128 layers: Liger peak 9032.3 MB   Fused peak 4424.3 MB   Memory saved 4608.0 MB
256 layers: Liger peak 17992.3 MB   Fused peak 8776.3 MB   Memory saved 9216.0 MB
```