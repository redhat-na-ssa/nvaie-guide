# Triton Inference Server

[NVIDIA Dynamo-Triton](https://developer.nvidia.com/triton-inference-server) (a.k.a Triton Inference Server) is available with enterprise-grade support, security, stability, and manageability with [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/get-started/). NVIDIA Dynamo will be included in NVIDIA AI Enterprise for production inference in a future release.

## Overview

This mini-workshop should give a hands-on introduction to the Triton model server

Workflow.
1. Deploy Triton on Openshift
1. Verify the model server is up and running.
1. Copy a sample model repository to the model storage.
1. Confirm the model has been loaded, is healthy and ready to accept requests.
1. Make an inference
1. Train a new model and upload the new version to the model's storage.
1. Make an inference to the new version of the model.

### Requirements

1. An [Openshift cluster with an NVIDIA GPU](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.ocp4-demo-rhods-nvidia-gpu-aws.prod&utm_source=webapp&utm_medium=share-link)

### Deployment
- Deploy triton using [Cory's automation](https://github.com/redhat-na-ssa/demo-triton-yolo.git). It should deploy the server along with a service and a route for inference and metrics.
The deployer will pull the [Triton server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) from NVIDIA's container catalog.
```bash
oc apply -k https://github.com/redhat-na-ssa/demo-triton-yolo/gitops/overlays/triton
```

### Verify
- Examine the pod logs.
```bash
POD=$(oc get pod -l app=triton-server -o custom-columns=POD:.metadata.name --no-headers)

oc logs ${POD} | grep HTTP
```
Example output
```bash
I0521 18:49:00.663425 23 http_server.cc:4755] "Started HTTPService at 0.0.0.0:8000"
```

- Get the route
```bash
export HOST="https://"$(oc get route triton-server -o jsonpath='{.spec.host}')
```

Triton supports the [kserve api](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) and
[extentions](https://github.com/triton-inference-server/server/tree/main/docs/protocol)
 for predictive models and the [OpenAI api](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client_guide/openai_readme.html) for LLMs.

- Check the health endpoint for an `HTTP/1.1 200 OK`

```bash
curl -vk $HOST/v2/health/ready
```

Look for the following message in the output
```console
< HTTP/1.1 200 OK
```

### Upload the sample model repository

- The following model repository is provided.

```console
models
└── lr               (model name)
    ├── 1            (version)
    │   └── model.pt (PyTorch model file)
    └── config.pbtxt (configuration)
```

- Copy this directory into the Triton container storage.
```bash
POD=$(oc get pod -l app=triton-server -o custom-columns=POD:.metadata.name --no-headers)
oc cp models/lr ${POD}:/models
```

- Restart Triton (don't think this is necessary since polling is enabled)
```bash
oc delete pod ${POD}
```

### Confirm the model is ready to accept requests

- Model Status Endpoint checks
```bash
curl -s $HOST/v2/models/lr | jq
```
```bash
curl -s $HOST/v2/models/lr/config | jq
```

- Model Status checks

```bash
curl -s -X POST $HOST/v2/repository/index | jq
```

Get the configuration for version 2 of the "lr" model.
```bash
curl -s ${HOST}/v2/models/lr/versions/2/config | jq
```

- Finally, make an inference with `curl`
```bash
curl -s -X POST -H "Content-Type: application/json" -d '{ "inputs": [ { "name": "input_name", "shape": [1], "datatype": "FP32", "data": [2.0] } ] }' ${HOST}/v2/models/lr/infer | jq .
```
Example output
```json
{
  "model_name": "lr",
  "model_version": "1",
  "outputs": [
    {
      "name": "output_name",
      "datatype": "FP32",
      "shape": [
        1
      ],
      "data": [
        3.989671230316162
      ]
    }
  ]
}
```

### Make an inference with a Python client
```bash
python 02-inference-triton.py
```

Sample output
```console
INFO:root:REST inference response content = {"model_name":"lr","model_version":"1","outputs":[{"name":"output_name","datatype":"FP32","shape":[1],"data":[4.999999523162842]}]}
```

Get the stats
```bash
curl ${HOST}/v2/models/lr/versions/1/stats
```

- Train and copy version 2 of the model. The script should create a new model directory.

```bash
python 01-train-save-model.py
```

- Copy it to the Triton pod and watch the logs. Version 2 of the model
should get loaded.

Example output:
```console
I0602 21:22:32.942157 23 server.cc:376] "Polling model repository"
I0602 21:22:32.942499 23 model_config_utils.cc:753] "Server side auto-completed config: "
name: "lr"
platform: "pytorch_libtorch"
version_policy {
  all {
  }
}
input {
  name: "input_name"
  data_type: TYPE_FP32
  dims: 1
}
output {
  name: "output_name"
  data_type: TYPE_FP32
  dims: 1
}
default_model_filename: "model.pt"
backend: "pytorch"
```
Example log output
```console
...
...
I0602 21:22:32.942582 23 model_lifecycle.cc:473] "loading: lr:1"
I0602 21:22:32.942633 23 model_lifecycle.cc:473] "loading: lr:2"
I0602 21:22:32.959919 23 model_lifecycle.cc:849] "successfully loaded 'lr'"
```

- Make an inference of the version 2 model.

The inference url takes the form `/v2/models/<model_name>/versions/<version>/infer`

```bash
curl -s -X POST -H "Content-Type: application/json" -d '{ "inputs": [ { "name": "input_name", "shape": [1], "datatype": "FP32", "data": [3.0] } ] }' ${HOST}/v2/models/lr/versions/2/infer | jq .
```

Example output

```console
{
  "model_name": "lr",
  "model_version": "2",
  "outputs": [
    {
      "name": "output_name",
      "datatype": "FP32",
      "shape": [
        1
      ],
      "data": [
        9.214284896850586
      ]
    }
  ]
}
```

- Bonus exercise

Modify the `02-inference-triton.py` script to inference version 2 of the model.

The output should resemble the following:
```console

```

### RHEL9

Do this once.
```bash
sudo setsebool -P container_use_devices true
```

```bash
podman run --gpus 0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/models:/models:z nvcr.io/nvidia/tritonserver:25.04-py3 tritonserver --model-store=/models --strict-model-config=false --log-verbose=1
```
