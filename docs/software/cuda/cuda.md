# CUDA Mini-Workshop

## Overview

CUDA is a platform and programming model for NVIDIA CUDA-enabled GPUs. The platform exposes GPUs for general purpose computing. CUDA provides C/C++ language extension and APIs for programming and managing GPUs.

CUDA is a foundational API that is closest to the GPU hardware. 
Higher level AI/ML frameworks such as PyTorch and Tensorflow
are built on top of CUDA to achieve acceleration on NVIDIA GPU platforms.

##### Compiling and running a few simple code examples.

A mini-workshop to learn about compiling and running simple CUDA programs on RHEL.

- Order the Base Red Hat AI Inference Server (RHAIIS) from the demo catalog.
- Perform the following to prepare the system to compile and run CUDA programs.
  - `export PATH=$PATH:/usr/local/cuda/bin`
  - Even better, modify your `~/.bashrc`

- Install the `ImageMagick-c++-devel` rpm

`sudo yum install ImageMagick-c++-devel -y`

- Clone https://github.com/harrism/nsys_easy
	- Move the `nsys_easy` script into a directory contained in $PATH
	- `mkdir $HOME/.local/bin` is a good option
- Compiling and running programs

`cd src`

`nvcc add_cuda.cu -o add_cuda`

`add_cuda`

- Run the profiler.

```bash
nsys_easy <executable>
```

Containers may core dump cuda programs if the cuda versions between the container and host are not 
the same.

```bash
podman run -it --rm -v $(pwd):/scratch:z nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04 bash
```

###### References

[CUDA overview from SC2011](https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)

[CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/)

https://developer.nvidia.com/blog/even-easier-introduction-cuda/

https://github.com/harrism/mini-nbody.git

