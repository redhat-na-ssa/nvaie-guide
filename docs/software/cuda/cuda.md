# CUDA Mini-Workshop

An introductory workshop to learn about compiling, running and tuning simple CUDA programs on RHEL.

## Overview

CUDA is a platform and programming model for NVIDIA CUDA-enabled GPUs that exposes the hardware for general 
purpose computing use cases. It provides a C/C++ language extension and APIs for programming and managing GPU
resources.

The CUDA API/SDK is a vector programming model that is close to the GPU hardware.
Higher level AI/ML frameworks such as PyTorch and Tensorflow
are built on top of CUDA to achieve AI/ML training and inference acceleration.

#### Why learn CUDA basics?

You don't need to know CUDA to sell Openshift or train ML models, but it will give you
a better appreciation for how GPUs work and the advantages they bring to AI engineers.
Besides, its nerdy fun.

#### Prerequisites

- Order the **Base Red Hat AI Inference Server (RHAIIS)** from the demo catalog.
This is a RHEL9/GPU VM with most of the NVIDIA driver and CUDA stack installed.

- Configure `ssh` so you can login w/o being prompted for a password (i.e. `ssh-copy-id`). This will
save you some typing.

- There are a fews ways to develop, build and run CUDA programs in this workshop.

1. Install `vscode` on the RHEL VM and use tunneling to connect from a web-based vscode session or
directly from your laptop.

```bash
curl -L -o code.tar.gz 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64
tar zxvf code.tar.gz
mkdir $HOME/.local/bin
mv code $HOME/.local/bin
code tunnel --accept-server-license-terms --name=<REPLACE_with_a_unique_name_or_your_initials>
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
	- `$HOME/.local/bin` is a good option

#### Complete the following exercises:

- Begin by running `make` to build all of the programs.

```bash
cd src
make
```

1. Run `./01-mycudadevice` to display the device properties. How much memory does your GPU have?

`./01mycudadevice`

- Search for TODO in the `01-mycudadevice.cu` source file and modify the program 
to print out the maximum number of threads per multiproccesor.
See the [CUDA API docs for the correct property to retrieve.](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp) 

- What is the theoretical maximum number of simultaneous threads that can execute on your device?

2. Run a simple vector add program to confirm the GPU can run CUDA programs.

```bash
./02-add_cuda_block
```

3. Complete the example image processing program.

- Begin by opening `03-rgb2gray.cu`
  - Complete TODO #1 by merging `snippets/convert.cu` into `03-rgb2gray.cu`
  - Run a `make` to confirm that there are no typos.
  - Complete TODO #2 by merging `snippets/call_kernel.cu` into `03-rgb2gray.cu`
  - Run a `make` to confirm that there are no typos.
  - If everything goes well the program should create a `gray.png` image that is visually correct.

4. Run the profiler and take note on the execution time of the `CUDA_KERNEL`.

```bash
nsys_easy ./03-rgb2gray
```

5. Now experiment with different block thread sizes and observe/compare the execution times.

- What is the optimal number of t for your device?

```bash
nsys_easy ./03-rgb2gray -t 8
```

#### End of Workshop


###### References

[CUDA overview from SC2011](https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)

[CUDA Tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/)

https://developer.nvidia.com/blog/even-easier-introduction-cuda/

https://github.com/harrism/mini-nbody.git

