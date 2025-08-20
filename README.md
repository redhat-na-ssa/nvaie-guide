# The Nvidia AI Enterprise Guide to the Galaxy

## About NVAIE

Nvidia AI Enterprise (NVAIE) is a collection of infrastructure and software tools provided by Nvidia to build a technical solution that can run predictive or generative AI inferencing.

NVAIE is licensed per GPU, includes Business Standard Support, and is purchased as an annual, multi-year, or hourly subscription (via Cloud Marketplaces).

In some cases, NVAIE is included, e.g NVAIE is included with the purchase of Nvidia DGX Systems.

To read the NVAIE components, go [here](https://docs.nvidia.com/ai-enterprise/release-6/6.0/getting-started/quick-start-guide.html#installing-nvidia-ai-enterprise-software-components).

To read the NVAIE Lifecycle Policy, go [here](https://docs.nvidia.com/ai-enterprise/lifecycle/latest/lifecycle-policy.html).

## About This Guide

This guide is intended for Red Hat Solution Architects who want a hands-on tour of NVAIE technical components running on Red Hat Platforms (RHEL and OpenShift).

Please complete the prerequisites first. After completing the prerequisites, you can jump straight into sections you are interested in.

## Guide

- [Prerequisites](docs/prereqs.md)
- Infrastructure
  1. [GPU Operator](docs/infra/gpu-operator.md)
  1. [Time Slice](docs/infra/gpu-timeslice.md)
  1. [MIG](docs/infra/gpu-mig.md)
  1. [NIM Operator](docs/infra/nim-operator.md)
  1. Network Operator
  1. vGPU
  1. Base Command Manager 
- Software
  1. [CUDA](docs/software/cuda/cuda.md)
  1. [PyTorch](docs/software/pytorch/pytorch.md)
  1. [TensorFlow](docs/software/tensorflow/tensorflow.md)
  1. [TensorRT](docs/software/tensorrt/README.md)
  1. [Triton Inference Server](docs/software/triton/README.md)
  1. AI-Dynamo
  1. [NeMo Framework](docs/software/nemo.md)
  1. [NIM](docs/software/nim.md)
  1. RAPIDS and RAPIDS Accelerator for Apache Spark 
  1. TAO (Computer Vision)
  1. [Riva (Speech-to-Text)](docs/software/riva.md)
  1. DeepStream (Video)
  1. Clara Parabricks (Genomics)
  1. MONAI (Medical Imaging)
- Appendix
  1. [KAI Scheduler](docs/appendix/kai.md)
  1. Kserve with Triton Inference Server
  1. Kserve with NIM

