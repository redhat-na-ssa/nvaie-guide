# PyTorch

[PyTorch](https://pytorch.org) is an optimized tensor library for deep learning using GPUs and CPUs.
It is typically used for pre-training, transfer learning and fine tuning of models for both
predictive and generative use cases. 

NVIDIA provides a [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to 
make things a bit easier for developers.

### Getting started with the NVIDIA PyTorch container.
#### RHEL9.5

The NVIDIA PyTorch container is a pre-configured container that provides a ready-to-use environment for deploying PyTorch applications. 

1. Prerequisites:
 - a. Install `podman`.
 - b. Check for a working NVIDIA CUDA stack:
```bash
nvidia-smi --list-gpus
```
Example output:
```
GPU 0: NVIDIA L4 (UUID: GPU-8b456645-6ed1-cf79-714e-19b2657eca53)
```
2. Set the TAG and IMAGE variables and run the NVIDIA PyTorch container to check for `cuda` to be printed last which
indicates that PyTorch found a GPU.

```bash
TAG=25.03-py3
IMAGE=nvcr.io/nvidia/pytorch:${TAG}
```
```bash
podman run --rm -it --name pytorch --security-opt=label=disable --device nvidia.com/gpu=all ${IMAGE} -- python -c 'import torch;print(torch.accelerator.current_accelerator())'
```
Example output:
```
=============
== PyTorch ==
=============

NVIDIA Release 25.03 (build 148941828)
PyTorch Version 2.7.0a0+7c8ec84
...
...
...

cuda
```
##### Training a model and making an inference with Python scripts.
This [PyTorch quickstart tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) is a nice
way to get started with some Python code. As a start, try the Python script from the [github repo](https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/quickstart_tutorial.py) on RHEL9.
```bash
git clone https://github.com/pytorch/tutorials.git
cd tutorials
TAG=25.03-py3
IMAGE=nvcr.io/nvidia/pytorch:${TAG}

podman run --rm -it --name pytorch -v $(pwd)/beginner_source/basics:/basics:z --security-opt=label=disable --device nvidia.com/gpu=all ${IMAGE} -- python /basics/quickstart_tutorial.py
```
Example Output:
```
=============
== PyTorch ==
=============
...
...
...
Using cuda device
...
...
...
Epoch 5
-------------------------------
loss: 1.067148  [57664/60000]
Test Error:
 Accuracy: 65.0%, Avg loss: 1.087852

Done!
Saved PyTorch Model State to model.pth
Predicted: "Ankle boot", Actual: "Ankle boot"
```

##### Training a model and making an inference with Jupyter Notebooks.

The NVIDIA Pytorch container includes the Jupyter Lab software. To use this, make sure your virtual machine allows incoming
traffic on port 8888/tcp.

###### Run the PyTorch container and launch `jupyter-lab` at start up.

Download the sample notebook first.
```bash
mkdir notebooks
cd notebooks
wget https://pytorch.org/tutorials/_downloads/af0caf6d7af0dda755f4c9d7af9ccc2c/quickstart_tutorial.ipynb
```

Now run the container and watch for a token to appear in the logs, you'll need this to login to Jupyter.

```bash
podman run -p8888:8888 --rm -it --name pytorch -v $(pwd)/:/notebooks:z --security-opt=label=disable --device nvidia.com/gpu=all ${IMAGE} -- jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --notebook-dir=/notebooks
```

```
...
...
...
Copy and paste one of these URLs:
        http://hostname:8888/lab?token=f6dc2cf509b46b44fa8492a50b8e8e5c51f55c591ecea67f

```
Replace `hostname` with your hostname and visit this URL with a web browser and you should see the 
Jupyter login page.
