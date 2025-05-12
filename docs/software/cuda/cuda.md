# CUDA

## Overview

CUDA is a platform and programming model for NVIDIA CUDA-enabled GPUs. The platform exposes GPUs for general purpose computing. CUDA provides C/C++ language extension and APIs for programming and managing GPUs.

CUDA is a foundational API that is closest to the GPU hardware. 
Higher level AI/ML frameworks such as PyTorch and Tensorflow
are built on top of CUDA to achieve acceleration on NVIDIA GPU platforms.

The CUDA compiler is included on NVIDIA's [CUDA](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags), Pytorch and Tensorflow containers.

##### Compiling and running a few simple code examples.

```bash
cd docs/software/cuda
TAG=12.9.0-cudnn-devel-ubi9
IMAGE=nvcr.io/nvidia/cuda:${TAG}
podman run --rm -it --name cuda -v $(pwd)/:/cuda:z --security-opt=label=disable --device nvidia.com/gpu=all ${IMAGE} -- bash
```

```bash
cd /cuda
nvcc vector_add.cu -o vector_add
nvcc vector_add.cu -o vector_add_thread
nvcc vector_add_grid.cu -o vector_add_grid
```

Observe the performance of each program.
```bash
./vector_add
./vector_add_thread
./vector_add_grid
```

###### References

[CUDA overview from SC2011](https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)

[CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/)

