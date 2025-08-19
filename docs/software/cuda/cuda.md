# Hands-On CUDA Mini-Workshop
A mini-workshop to learn about compiling and running simple CUDA programs on RHEL.

## Work in Progress

### Overview

CUDA is a platform and programming model that exposes NVIDIA GPUs for general purpose computing. 
It provides a C/C++ language extension and APIs for programming and managing GPU resources.

CUDA is a foundational API that is closest to the GPU hardware and is often used in HPC use cases.
Higher level AI/ML frameworks such as PyTorch and Tensorflow
are built on top of CUDA to achieve acceleration of AI/ML training and inferencing on NVIDIA GPU platforms.

##### System setup

- Order the Base Red Hat AI Inference Server (RHAIIS) from the demo catalog.
- Perform the following to prepare the system to compile and run CUDA programs.
  - `export PATH=$PATH:/usr/local/cuda/bin`
  - Even better, modify your `~/.bashrc`

```console
mkdir ~/.bashrc.d
```
- Add the following to your `$HOME/.bashrc.d/local` file.

```console
# Add CUDA to the PATH
if ! [[ "$PATH" =~ "/usr/local/cuda/bin" ]]
then
    PATH="/usr/local/cuda/bin:$PATH"
fi
export PATH
```

Test

```bash
source $HOME/.bashrc.d/local
echo $PATH
```

- Install a few needed rpms.

`sudo yum install ImageMagick-c++-devel install nsight-systems-2024.6.2 bc -y`

- Clone https://github.com/harrism/nsys_easy
	- Move the `nsys_easy` script into a directory contained in $PATH
	- `mkdir $HOME/.local/bin` is a good option

#### Exercises to complete.

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

