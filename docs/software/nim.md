# NIM (Nvidia Inference Microservices)

> TODO: Preamble

> TODO: Note L4 is not technically supported, see [supported GPUs](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#gpus)

> [!IMPORTANT]
> Make sure you have user workload monitoring enabled, see the [Prerequisites](docs/prereqs.md).
> Make sure you have your Nvidia API key, see the [Prerequisites](docs/prereqs.md).

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

> TODO: Add description of model profiles

We will use the [list-model-profiles](https://docs.nvidia.com/nim/large-language-models/latest/utilities.html#list-available-model-profiles) utility to determine which model profiles exist.

Run a job using the `list-model-profiles` utility:

```bash
oc create -f configs/software/nim/meta-profiles.yaml 
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

> TODO: Explain the profiles we selected

Look at the NIM cache file:

```bash
cat configs/software/nim/meta-cache.yaml
```

Notice the following:

- [ ] `volumeAccessMode` is set to `ReadWriteOnce`
- [ ] `model.profiles` includes the profiles `7cc...`

While Nvidia recommends RWX volume access mode, we will use RWO for demonstration purposes. Change this to `ReadWriteMany` if you have an appropriate storage class that supports RWX.

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

Smoke test

```bash
NIM_META_URL=$(oc get route -n nim meta-llama3-8b-instruct --template='http://{{.spec.host}}')

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
NIM_META_URL=$(oc get route -n nim meta-llama3-8b-instruct --template='http://{{.spec.host}}')

curl $NIM_META_URL/v1/metrics
```

One of the metrics is `gpu_cache_usage_perc`. We'll use that in the autoscaling configuration.

> TODO: Add note prometheus adapter deprecated in OpenShift

Install CMA Operator

> TODO: Add steps

Create KedaController

```bash
oc create -f configs/software/nim/keda-controller.yaml
```

Configure authentication for KEDA trigger

```bash
oc create -f configs/software/nim/thanos-sa-secret.yaml
```

Create trigger

```bash
oc create -f configs/software/nim/keda-trigger.yaml -n nim
```

Create role binding

```bash
oc create -f configs/software/nim/thanos-role-rolebinding.yaml -n nim
```

Note annotation that pauses the ScaledObject

```bash
cat configs/software/nim/meta-scaledobject.yaml
```

Create ScaledObject with scaling set to PAUSE

```bash
oc create -f configs/software/nim/meta-scaledobject.yaml 
```

> TODO: Update NIM hpa to monitor external metric

> TODO: Update NIM with autoscaling

Notice we are using a very low value `0.01`. This should be `0.5` (50%) but we're using a really low number to force the autoscale for demo purposes.

Smoke test

```bash
while true; do sleep 1; curl -X "POST" \
 $NIM_META_URL/v1/chat/completions \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'; done 
```

Delete NIM service:

```bash
oc -n nim delete nimservice meta-llama3-8b-instruct
```

Delete NIM cache:

```bash
oc -n nim delete nimcache meta-llama-3-8b-instruct
```

## NIM Pipelines

> TODO

## NeMo

> TODO: Bring a NeMo custom model from the NeMo section and deploy it as a NIM

