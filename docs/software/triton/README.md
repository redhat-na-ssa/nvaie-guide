# Triton Inference Server (work in progress)

[NVIDIA Dynamo-Triton](https://developer.nvidia.com/triton-inference-server) (a.k.a Triton Inference Server) is available with enterprise-grade support, security, stability, and manageability with [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/get-started/). NVIDIA Dynamo will be included in NVIDIA AI Enterprise for production inference in a future release.

### Requirements

1. An [Openshift cluster with an NVIDIA GPU](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.ocp4-demo-rhods-nvidia-gpu-aws.prod&utm_source=webapp&utm_medium=share-link)

### Deployment
- Deploy triton using [Cory's automation](https://github.com/redhat-na-ssa/demo-triton-yolo.git). It should deploy the server along with a service and a route for inference and metrics.
The deployer will pull the [Triton server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver) from NVIDIA's container catalog.
```bash
oc apply -k https://github.com/redhat-na-ssa/demo-triton-yolo/gitops/overlays/triton
```
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
HOST="https://"$(oc get route triton-server -o jsonpath='{.spec.host}')
```

#### Using the API

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

- Restart Triton
```bash
oc delete pod ${POD}
```

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

Get the stats
```bash
curl ${HOST}/v2/models/lr/versions/1/stats
```

- Train and copy version 2 of the model

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

- Inference version 2
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

RHEL9

Do this once.
```bash
sudo setsebool -P container_use_devices true
```

```bash
podman run --gpus 0 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/models:/models:z nvcr.io/nvidia/tritonserver:25.04-py3 tritonserver --model-store=/models --strict-model-config=false --log-verbose=1
```
