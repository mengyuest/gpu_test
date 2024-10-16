# Pytorch (GPU/CPU/MPS) runtime test

> Test pytorch-related tasks in different devices (cpu, gpu, mps...)


## Prerequisite
For basic tests: `conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia`

For CNN test: `conda install torchvision==0.14.1 -c pytorch -c nvidia`

For GNN test: `conda install pyg -c pyg`

## Usage
> `usage: gpu_profiling.py [-h] [--device {cpu,gpu,mps,default}] [--tests TESTS [TESTS ...]]`

### Example-1: Run all tests on the default device (use GPU when possible)

`python gpu_profiling.py`


### Example-2: Run all tests on GPU

`python gpu_profiling.py --device gpu`

### Example-3: Run matrix multiplication test on CPU

`python gpu_profiling.py --device cpu --tests mm`

### Example-4: Run MLP and CNN tests on MPS 

`python gpu_profiling.py --device mps --tests mlp cnn`