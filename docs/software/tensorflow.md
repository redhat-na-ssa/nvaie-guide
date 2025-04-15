# Tensorflow

[Tensorflow](https://tensorflow.org) is an optimized tensor library for deep learning using GPUs and CPUs.
It is typically used for pre-training, transfer learning and fine tuning of models for both
predictive and generative use cases. 

NVIDIA provides a [Tensorflow container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) to 
make things a bit easier for developers.

### Getting started with the NVIDIA Tensorflow container.
#### RHEL9.5

The NVIDIA Tensorflow container is a pre-configured container that provides a ready-to-use environment for deploying Tensorflow applications. 

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
2. Set the TAG and IMAGE variables then run the NVIDIA Tensorflow container. The last line should indicate that a GPU was found.
```bash
TAG=25.02-tf2-py3
IMAGE=nvcr.io/nvidia/tensorflow:${TAG}
```
```bash
podman run --rm -it --name Tensorflow --security-opt=label=disable --device nvidia.com/gpu=all ${IMAGE} -- python -c 'import tensorflow;print(tensorflow.config.list_logical_devices())'
```
Example output:
```
================
== TensorFlow ==
================

NVIDIA Release 25.02-tf2 (build 143088766)
TensorFlow Version 2.17.0
...
...
...
20581 MB memory:  -> device: 0, name: NVIDIA L4, pci bus id: 0000:31:00.0, compute capability: 8.9
[LogicalDevice(name='/device:CPU:0', device_type='CPU'), LogicalDevice(name='/device:GPU:0', device_type='GPU')]
```
##### Training a model
As a start, run the trivial Tensorflow Python script that is built into the container.

```bash
podman run --rm -it --name Tensorflow --security-opt=label=disable --device nvidia.com/gpu=all ${IMAGE} -- python nvidia-examples/cnn/trivial.py
```
Example output:
```
...
...
...
epoch: 0 time_taken: 22.9
300/300 - 23s - loss: 6.9190 - top1: 0.0082 - top5: 0.0360 - 23s/epoch - 76ms/step
```

This [Tensorflow quickstart tutorial](https://Tensorflow.org/tutorials/beginner/basics/quickstart_tutorial.html) is a nice
way to get started with some Python code.

#### Openshift 4.18
