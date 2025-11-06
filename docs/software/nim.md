# NIM (Nvidia Inference Microservices)

View the [supportability page](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#) to view supported GPUs and LLMs for NIM.

> [!IMPORTANT]
> Configure User Workload Monitoring and Nvidia API Key in [Prerequisites](../prereqs.md).\

### Deploy on RHEL

#### Prerequisites

- RHEL9.5 AWS instance type `g6e.xlarge` with a CUDA runtime stack.

```bash
export NGC_API_KEY=<your_ngc_api_key>

podman login nvcr.io
Username: $oauthtoken
Password: <NGC_API_KEY>
Login Succeeded!

export LOCAL_NIM_CACHE=~/.cache/nim
export HF_TOKEN=<your_huggingface_token>
mkdir -p "$LOCAL_NIM_CACHE"
sudo chcon -R system_u:object_r:container_home_t:s0 .cache/nim
chmod 777 .cache/nim 
```

Launch the NIM container. These are large images and could take several
minutes to download and become ready.

```bash
podman run -it --rm --security-opt=label=disable --device nvidia.com/gpu=all --name=nim-server -e NGC_API_KEY  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache:z" -p 8000:8000 nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
```

Example output
```console
===========================================
== NVIDIA Inference Microservice LLM NIM ==
===========================================

NVIDIA Inference Microservice LLM NIM Version 1.13.1
Model: meta-llama/llama-3.1-8b-instruct
...
...
...
INFO 11-06 17:50:16 [__init__.py:248] Automatically detected platform cuda.
...
...
...
INFO 2025-11-06 17:54:19.817 api_server.py:235] Available routes are:
INFO 2025-11-06 17:54:19.817 api_server.py:243] Route: /openapi.json, Methods: HEAD, GET
INFO 2025-11-06 17:54:19.817 api_server.py:243] Route: /docs, Methods: HEAD, GET
INFO 2025-11-06 17:54:19.817 api_server.py:243] Route: /docs/oauth2-redirect, Methods: HEAD, GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/chat/completions, Methods: POST
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/completions, Methods: POST
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /experimental/ls/inference/chat_completion, Methods: POST
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /experimental/ls/inference/completion, Methods: POST
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/health/ready, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/health/live, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/models, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/metadata, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/version, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/health/live, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/health/ready, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/metrics, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/license, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/metadata, Methods: GET
INFO 2025-11-06 17:54:19.818 api_server.py:243] Route: /v1/manifest, Methods: GET
INFO 2025-11-06 17:54:19.885 server.py:82] Started server process [107]
INFO 2025-11-06 17:54:19.886 on.py:48] Waiting for application startup.
INFO 2025-11-06 17:54:20.130 on.py:62] Application startup complete.
INFO 2025-11-06 17:54:20.131 server.py:214] Uvicorn running on http://0.0.0.0:8000
```

Perform a basic health check.
```
curl http://0.0.0.0:8000/v1/models | jq
```
Example output
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/llama-3.1-8b-instruct",
      "object": "model",
      "created": 1762451767,
      "owned_by": "system",
      "root": "meta-llama/llama-3.1-8b-instruct",
      "parent": null,
      "max_model_len": 131072,
      "permission": [
        {
          "id": "modelperm-3d30ff4eb1ac447fa514ac085459d831",
          "object": "model_permission",
          "created": 1762451767,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

Make an inference
```bash
curl -s -X 'POST' \
  'http://0.0.0.0:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      },
      {
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      },
      {
        "role":"user",
        "content":"Can you write me a song?"
      }
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": false,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }' | jq
  ```
  
  Example output
  ```json
{
  "id": "chatcmpl-3f633fd56e4f447fa61c210bfbb1e999",
  "object": "chat.completion",
  "created": 1762451809,
  "model": "meta-llama/llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": "I'd be happy to write a song for you. Before I start,",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 72,
    "total_tokens": 87,
    "completion_tokens": 15,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "kv_transfer_params": null
}
  ```

[NVIDIA Reference Doc](https://docs.nvidia.com/nim/visual-genai/latest/getting-started.html)


### Deploy on Openshift
> Configure [GPU Operator](../infra/gpu-operator.md).\
> Configure [NIM Operator](../infra/nim-operator.md).

Let's create a namespace `nim` to work in.

Create a new namespace:

```bash
oc create -f configs/software/nim/nim-ns.yaml
```

Create secrets associated with your API Key:

```bash
oc -n nim create secret docker-registry ngc-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=$NGC_API_KEY
oc -n nim create secret generic ngc-api-secret --from-literal=NGC_API_KEY=$NGC_API_KEY
```

## Caching Models

The first step in NIM is to cache (i.e. download) the model.

Before we can create our model cache, we have to specify which model profile we want to cache.

The model profile defines which model engine NIM can use for the Large Language Model we are deploying. View this [page](https://docs.nvidia.com/nim/large-language-models/latest/profiles.html#model-profiles) for more information on model profiles.

We will use the [list-model-profiles](https://docs.nvidia.com/nim/large-language-models/latest/utilities.html#list-available-model-profiles) utility to determine which model profiles exist.

Run a job using the `list-model-profiles` utility:

```bash
oc create -n nim -f configs/software/nim/meta-profiles.yaml 
```

Wait for job to complete:

```bash
oc wait -n nim --for=condition=complete job nim-profile-job --timeout=100s
```

```text
job.batch/nim-profile-job condition met
```

View the logs of the job:

```bash
oc logs -n nim $(oc get pod -n nim -l job-name=nim-profile-job -o jsonpath='{.items[0].metadata.name}')
```

There are two primary profiles that are compatible: the vLLM engine (193...) and TensorRT (7cc...). We'll use the TensorRT engine in this example.

Look at the NIM cache file:

```bash
cat configs/software/nim/meta-cache.yaml
```

Notice the following:

- [ ] `volumeAccessMode` is set to `ReadWriteMany`
- [ ] `model.profiles` includes the profiles `7cc...`

The RWX volume access mode allows us to create a single NIM cache holding the model (a PVC) that can be mounted on two separate nodes/machines as the NIMService autoscales up.

Deploy a NIM cache for Meta Llama-3.1-8b-Instruct:

```bash
oc create -n nim -f configs/software/nim/meta-cache.yaml
```

Wait for the NIM cache to be ready:

```bash
oc wait -n nim --for=condition=complete job meta-llama3-8b-instruct-job --timeout=100s
```

```text
job.batch/meta-llama3-8b-instruct-job condition met
```

You now have a Persistent Volume with meta-llama3-8b-instruct downloaded.

## NIM Services

Deploy a NIM service for Meta Llama-3.1-8b-Instruct using the cache:

```bash
oc create -n nim -f configs/software/nim/meta-service.yaml
```

Wait for the NIM service to be ready. Note this can take some time:

```bash
oc rollout status deploy/meta-llama3-8b-instruct -n nim --timeout=600s
```

```text
deployment "meta-llama3-8b-instruct" successfully rolled out
```

Expose the service with a route:

```bash
oc expose -n nim svc meta-llama3-8b-instruct
```

Grab a reference to the route:

```bash
NIM_META_URL=$(oc get route -n nim meta-llama3-8b-instruct --template='http://{{.spec.host}}')
```

Smoke test

```bash
curl -X "POST" \
 $NIM_META_URL/v1/chat/completions \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```text
{"id":"chat-7badc43d2979411c86b70937861ae104","object":"chat.completion","created":1743615706,"model":"meta/llama-3.1-8b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I assist you today?"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":12,"total_tokens":21,"completion_tokens":9},"prompt_logprobs":null}
```

The NIM service exposes a variety of metrics. Take a look:

```bash
curl $NIM_META_URL/v1/metrics
```

One of the metrics is `gpu_cache_usage_perc`. We'll use that in the autoscaling configuration.

### Autoscaling

Scale the machineset to allow NIM to autoscale to 2 replicas:

```bash
oc scale machineset gpu-machineset -n openshift-machine-api --replicas=2
```

NIM Services provide a specification to enable horizontal pod autoscaling (HPA). The Nvidia [documentation](https://docs.nvidia.com/nim-operator/latest/service.html#prerequistes-hpa) uses a Prometheus Adapter to create custom metrics for NIM HPA.

However, OCP removed the Prometheus Adapter starting in [v4.8](https://docs.openshift.com/container-platform/4.8/release_notes/ocp-4-8-release-notes.html#ocp-4-8-hpa-prometheus).

Instead, we will leverage the [Custom Metrics Autoscaler (CMA)](https://docs.redhat.com/en/documentation/openshift_container_platform/4.17/html/nodes/automatically-scaling-pods-with-the-custom-metrics-autoscaler-operator#nodes-cma-autoscaling-custom) based on KEDA to create external metrics for NIM HPA.

Start by installing the CMA Operator:

```bash
oc create -f configs/software/nim/cma-operator.yaml
```

Create `KedaController`:

```bash
oc create -f configs/software/nim/keda-controller.yaml
```

Configure service account authentication for KEDA trigger:

```bash
oc create -n nim -f configs/software/nim/thanos-sa-secret.yaml
```

Create role binding for KEDA trigger service account:

```bash
oc create -n nim -f configs/software/nim/thanos-role-rolebinding.yaml
```

Create the KEDA trigger authentication resource:

```bash
oc create -n nim -f configs/software/nim/keda-trigger.yaml
```

Now it's time to create a `ScaledObject`. This is a custom resource that defines autoscaling parameters and the scaling metric to monitor.

In a typical use case, you would deploy a `ScaledObject` and it would create a HPA for you.

In this use case, we need KEDA to define the scaling metric **but not create the HPA** because NIM will do that instead.

To do this, we have to add two annotations `paused-replicas` and `paused`. Both annotations are required; `paused-replicas` forces the `ScaledObject` to create the scaling metric and `paused` prevents the KEDA HPA from being deployed.

Note the two annotations we added to the `ScaledObject`:

```bash
cat configs/software/nim/keda-meta-scaledobject.yaml | grep annotation -A 2
```

```text
  annotations:
    autoscaling.keda.sh/paused-replicas: '1'
    autoscaling.keda.sh/paused: 'true'
```

Create `ScaledObject`:

```bash
oc create -n nim -f configs/software/nim/keda-meta-scaledobject.yaml 
```

Verify the KEDA external metric is created:

```bash
oc get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/nim/s0-prometheus?labelSelector=scaledobject.keda.sh%2Fname%3Dmeta-scaledobject"
```

```text
{"kind":"ExternalMetricValueList","apiVersion":"external.metrics.k8s.io/v1beta1","metadata":{},"items":[{"metricName":"s0-prometheus","metricLabels":null,"timestamp":"2025-04-07T14:31:11Z","value":"1m"}]}
```

Look at the HPA spec that monitors the KEDA external metric. Notice we are using a very low value `0.01`. This should be `0.5` (50%) but we're using a really low number to force the autoscale for demo purposes.

```bash
cat configs/software/nim/meta-service-hpa.yaml | grep External -B 1 -A 11
```

```text
      metrics:
      - type: External
        external:
          metric:
            name: s0-prometheus
            selector:
              matchLabels:
                scaledobject.keda.sh/name: meta-scaledobject
          target:
            type: Value
            value: '0.01'
```

Enable autoscaling in NIM:

```bash
oc apply -n nim -f configs/software/nim/meta-service-hpa.yaml
```

Bump up the KV cache by sending a more complex request:

```bash
curl -X "POST" \
 $NIM_META_URL/v1/chat/completions \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "stream": "True",
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Summarize Red Hat OpenShift"}]
  }'
```

Verify the HPA scaled the deployment:

```bash
oc get -n nim hpa meta-llama3-8b-instruct
```

```text
NAME                      REFERENCE                            TARGETS   MINPODS   MAXPODS   REPLICAS   AGE
meta-llama3-8b-instruct   Deployment/meta-llama3-8b-instruct   18m/10m   1         2         2          
```

Note you may have to wait ~10 minutes before the next step if the GPU Operator is still installing the GPU drivers on your second GPU machine.

Inspect the second replica of the NIM service:

```bash
oc get pods -n nim -l app=meta-llama3-8b-instruct
```

```text
NAME                                       READY   STATUS             RESTARTS        AGE
meta-llama3-8b-instruct-xxxxxxxxxx-xxxxx   1/1
meta-llama3-8b-instruct-xxxxxxxxxx-xxxxx   1/1     
```

### Cleanup

Delete scaled object:

```bash
oc -n nim delete scaledobject --all
```

Delete NIM service:

```bash
oc -n nim delete nimservice meta-llama3-8b-instruct
```

Scale the machineset down:

```bash
oc scale machineset gpu-machineset -n openshift-machine-api --replicas=1
```

## NIM Pipelines

If you have multiple NIM services, you can deploy multiple NIM services and manage each individually as a `NIMService`.

Alternatively, you can *group* NIM services together as one resource called a `NIMPipeline`.

Let's demonstrate this with the [llama-3.2-nv-embedqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2) and [llama-3.2-nv-rerankqa-1b-v2](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2) models.

Scale the machineset to allow NIM to autoscale to 2 replicas:

```bash
oc scale machineset gpu-machineset -n openshift-machine-api --replicas=2
```

Create the caches for both models. We'll skip identifying a profile and simply download `-all` profiles.

```bash
oc create -n nim -f configs/software/nim/pipeline-cache.yaml
```

Wait for the NIM caches to be ready:

```bash
oc wait -n nim --for=condition=complete job nv-embedqa-e5-v5-job --timeout=100s
oc wait -n nim --for=condition=complete job nv-rerankqa-mistral-4b-v3-job --timeout=100s
```

View the caches:

```bash
oc get -n nim nimcache
```

```text
NAME                        STATUS   PVC                             AGE
meta-llama3-8b-instruct     Ready    meta-llama3-8b-instruct-pvc     
nv-embedqa-e5-v5            Ready    nv-embedqa-e5-v5-pvc            
nv-rerankqa-mistral-4b-v3   Ready    nv-rerankqa-mistral-4b-v3-pvc 
```

Note you may have to wait ~10 minutes before the next step if the GPU Operator is still installing the GPU drivers on your second GPU machine.

Deploy the models together as a `NIMPipeline`:

```bash
oc create -n nim -f configs/software/nim/pipeline.yaml
```

Check the pipeline status:

```bash
oc get -n nim nimpipeline
```

```text
NAME       STATUS   AGE
pipeline   Ready    
```

View the NIM services running:

```bash
oc get pods -l app.kubernetes.io/part-of=nim-service
```

```text
NAME                                      READY   STATUS    RESTARTS   AGE
nv-embedqa-e5-v5-xxxxxxxxxx-xxxxx         1/1     Running   0          
nv-rerank-mistral-4b-v3-xxxxxxxxx-xxxxx   1/1     Running   0          
```

### Cleanup

Delete the NIM pipeline:

```bash
oc delete -n nim nimpipeline pipeline
```

Scale the machineset down:

```bash
oc scale machineset gpu-machineset -n openshift-machine-api --replicas=1
```

