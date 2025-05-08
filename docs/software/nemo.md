# NeMo Framework (Neural Modules)

NeMo is a framework for building and deploying neural network models and is built on top of PyTorch.

The NeMo Framework supports Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), and Text-to-Speech (TTS) modalities.

## Setup

> [!IMPORTANT]
> Make sure you have configured [GPU Operator](../docs/infra/gpu-operator.md).

There are [three primary ways](https://github.com/NVIDIA/NeMo?tab=readme-ov-file#install-nemo-framework) of installing NeMo:

1. Pip, requires pip installing the `nemo_toolkit` library
1. NGC PyTorch Container, requires installing the `nemo_toolkit` library using a built-in bash script
1. NGC NeMo Container

We are going to deploy the NGC NeMo Container in OpenShift. Inside of the container, we will launch a Jupyter notebook to run NeMo tutorials.

View the NeMo container (`nvcr.io/nvidia/nemo`) in the deployment file:

```bash
cat configs/software/nemo/nemo-deployment.yaml | grep nvcr -B 2
```

```text
      containers:
      - name: nemo
        image: nvcr.io/nvidia/nemo:25.02
```

Also view the Jupyter Notebook we are launching in the deployment file:

```bash
cat configs/software/nemo/nemo-deployment.yaml | grep jupyter -B 1 -A 1
```

```text
        args: [
          "jupyter notebook --allow-root --ip 0.0.0.0 --port 8088 --no-browser --NotebookApp.token=''"
        ]
```

Create the NeMo deployment:

```bash
oc new-project nemo
oc adm policy add-scc-to-user anyuid -z default
oc create -f configs/software/nemo/nemo-deployment.yaml
```

Expose a route to the deployment:

```bash
oc expose deploy nemo-deployment
oc expose svc nemo-deployment
```

Open the route in your browser (make sure to use non-TLS route):

```bash
echo $(oc get route nemo-deployment -n nemo --template='http://{{.spec.host}}')
```

Open JupyterLab by navigating to `View` -> `Open JupyterLab`.

## Tutorials

### Large Language Models

The NeMo library provides the ability to pretrain, fine-tune, and hyperparameter tune a Large Language Model. You can view examples of these capabilities here:

> Note: `auto_configurator` is the tutorial for hyperparameter tuning

```text
https://github.com/NVIDIA/NeMo/tree/main/examples/llm
```

The main module that has the LLM capabilities is:

```text
from nemo.collections import llm
``` 

Let's run a tutorial that will optimize a HuggingFace model with Parameter Efficient Fine Tuning (PEFT) using the NeMo collections API.

Download the following iPython notebook and upload to JupyterLab:

```text
https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/automodel/peft.ipynb
```

Execute the cells in the notebook.

### Multimodal Models

There are a variety of tutorials for NeMo MultiModal models, which you can view here.

```text
https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/README.md
```

Let's run the DreamBooth tutorial. In this tutorial, you will finetune a pretrained diffusion model using sample images of dogs. After fine-tuning, the model will be able to generate images of those dogs using text.

> TODO: Need a more powerful GPU

Download the following iPython notebook and upload to JupyterLab:

```text
https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/DreamBooth%20Tutorial.ipynb
```

Execute the cells in the notebook.

> You can also check out the [NeVA](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/multimodal/mllm/neva.html) tutorial listed below, but this example requires a significant amount of data preparation in order to pretrain a MultiModal LLM.

```text
https://github.com/NVIDIA/NeMo/blob/main/tutorials/multimodal/NeVA%20Tutorial.ipynb
```

### Automatic Speech Recognition (ASR) and Text to Speech (TTS)

You can view tutorials for ASR and TTS here:

```text
https://github.com/NVIDIA/NeMo/tree/main/tutorials/asr
https://github.com/NVIDIA/NeMo/tree/main/tutorials/tts
```

Let's run a tutorial that combines ASR and TTS in one use case. In this tutorial, you will transcribe speech in Mandarin Chinese to text (using ASR), translate that text to English (using NLP), and then generate English audio of those translations (using TTS).

Download the following iPython notebook and upload to JupyterLab:

```text
https://github.com/NVIDIA/NeMo/blob/main/tutorials/AudioTranslationSample.ipynb
```

## Clean Up

```bash
oc delete all --all -n nemo
oc delete project nemo
```

## Additional References

For tutorials on the lower level building blocks of the NeMo collection, check out:

```text
https://github.com/NVIDIA/NeMo/blob/main/tutorials/00_NeMo_Primer.ipynb
```

For additional tutorials that can be run on Google Colab, check out:

```text
https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/tutorials.html
```
