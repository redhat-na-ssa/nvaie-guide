# PyTorch

[PyTorch](https://pytorch.org) is an optimized tensor library for deep learning using GPUs and CPUs.

NVIDIA provides a [certified PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) to 
make things a bit easier.

#### Getting started with the NVIDIA PyTorch container.
##### RHEL9.5
To deploy pytorch on RHEL 9.5, you can use the NVIDIA PyTorch container. Here are the steps to get started:

The NVIDIA PyTorch container is a pre-configured container that provides a ready-to-use environment for deploying PyTorch applications on OpenShift. To use it, you can follow these steps:


To get started with the NVIDIA PyTorch container on RHEL 9.5, follow these steps:

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
2. Set the TAG variable and run the NVIDIA PyTorch container to check for `cuda` to be printed last.
```bash
TAG=25.03-py3
```
```bash
podman run --rm -it --name pytorch --security-opt=label=disable --device nvidia.com/gpu=all nvcr.io/nvidia/pytorch:${TAG} -- python -c 'import torch;print(torch.accelerator.current_accelerator())'
```
Example output:
```
=============
== PyTorch ==
=============

NVIDIA Release 25.03 (build 148941828)
PyTorch Version 2.7.0a0+7c8ec84
Container image Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2014-2024 Facebook Inc.
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
Copyright (c) 2015      Google Inc.
Copyright (c) 2015      Yangqing Jia
Copyright (c) 2013-2016 The Caffe contributors
All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

GOVERNING TERMS: The software and materials are governed by the NVIDIA Software License Agreement
(found at https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the Product-Specific Terms for NVIDIA AI Products
(found at https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).

cuda
```

##### Openshift 4.18
