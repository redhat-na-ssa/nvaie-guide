# CUDA Mini-Workshop

A mini-workshop to learn about compiling and running simple CUDA programs on RHEL.

## Overview

CUDA is a platform and programming model for NVIDIA CUDA-enabled GPUs. The platform exposes GPUs for general purpose computing. CUDA provides C/C++ language extension and APIs for programming and managing GPUs.

CUDA is a foundational API that is closest to the GPU hardware. 
Higher level AI/ML frameworks such as PyTorch and Tensorflow
are built on top of CUDA to achieve acceleration on NVIDIA GPU platforms.

#### Why learn CUDA basics?

You don't need to know CUDA to sell Openshift or train ML models but it will give you
a better appreciation for how GPUs work and the advantages they bring to AI engineers.
Besides, its nerdy fun.

#### Prerequisites

- Order the **Base Red Hat AI Inference Server (RHAIIS)** from the demo catalog.
This is a RHEL9/GPU VM with most of the NVIDIA driver and CUDA stack installed.

- Configure `ssh` so you can login w/o being prompted for a password (i.e. `ssh-copy-id`). This will
save you some typing.

- There are 3 ways to develop, build and run CUDA programs.

1. Install `vscode` on the VM and use tunneling to connect from a web-based vscode session or
directly from your laptop.

```bash
curl -L -o code.tar.gz 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64
tar zxvf code.tar.gz
mv code $HOME/.local/bin
code tunnel --accept-server-license-terms --name=<a_unique_name_or_your_initials>
```

2. Run vscode from your laptop and `ssh` into the VM.

3. Be a hardcore Linux type and use `ssh` and `vim`.

- Perform the following to prepare the system to compile and run CUDA programs.
  - `export PATH=$PATH:/usr/local/cuda/bin`
  - Even better, modify your `~/.bashrc`

- Install a few extra rpms.

```bash
sudo yum install ImageMagick-c++-devel bc -y
```

- Clone https://github.com/harrism/nsys_easy
	- Move the `nsys_easy` script into a directory contained in $PATH
	- `mkdir $HOME/.local/bin` is a good option

##### Compiling and running a few simple code examples.

`cd src`

1. Run a simple vector add program to confirm the GPU can run CUDA programs.

```bash
make
./02-add_cuda_block.cu
```

2. Examine the device properties. How much memory does your GPU have?

`./01mycudadevice`

- Modify `./01mycudadevice.cu` to print out the maximum number of threads per multiproccesor.
See the [CUDA API docs for the correct property to retrieve.](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp) 

- What is the theoretical maximum number of simultaneous threads that can execute on your device?

3. Complete the image processing example program.

- Begin by opening `03-rgb2gray.cu`
  - Complete TODO #1 by merging `snippets/convert.cu` into `03-rgb2gray.cu`
  - Run a `make` to confirm that there are no typos.
  - Complete TODO #2 by merging `snippets/call_kernel.cu` into `03-rgb2gray.cu`
  - Run a `make` to confirm that there are no typos.
  - If everything goes well the program should create a `gray.png` image.

- Run the profiler and take note on the execution time of the `CUDA_KERNEL`

```bash
nsys_easy ./03-rgb2gray
```

- Now experiment with different thread sizes and observe/compare the execution times.

```bash
nsys_easy ./03-rgb2gray -t 8
```
```bash
for i in 1 2 4 8 16 32; do echo "# CUDA_KERNEL Threads = " $i "x" $i;nsys_easy ./03-rgb2gray -t $i; done | grep CUDA_KERNEL
```

#### End of Workshop

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

