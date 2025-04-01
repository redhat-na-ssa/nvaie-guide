# NIM (Nvidia Inference Microservices)

> TODO: Preamble

> Note L4 is not technically supported, see [supported GPUs](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#gpus)

> Note: Make sure you have exported your `NGC_API_KEY`, see Prerequisites.

> TODO: Link instrutions to set NGC credentials

```sh
export NGC_API_KEY=
oc -n nim create secret docker-registry ngc-secret --docker-server=nvcr.io --docker-username='$oauthtoken' --docker-password=$NGC_API_KEY
oc -n nim create secret generic ngc-api-secret --from-literal=NGC_API_KEY=$NGC_API_KEY
```

## Caching Models

The first step in NIM is to cache (i.e. download) the model.

Create a new namespace

```sh
oc create -n -f configs/software/nim/nim-ns.yaml
```

Look at the NIM cache file

```sh
cat configs/software/nim/meta-cache.yaml
```

Notice the `volumeAccessMode` is set to `ReadWriteOnce`. Change this to `ReadWriteMany` if you have an appropriate storage class that supports RWX.

While Nvidia recommends RWX volume access mode, we will use RWO for demonstration purposes.

> TODO: Is there an easy way to try RWX access?

Deploy a NIM cache for Meta Llama-3.1-8b-Instruct

```sh
oc create -n nim -f configs/software/nim/meta-cache.yaml
```

Wait for the NIM cache to be ready

```sh
oc wait -n nim --for=jsonpath='{.spec.status}'='Ready' nimcache meta-llama3-8b-instruct --timeout=300s
```

## NIM Services

Deploy a NIM service for Meta Llama-3.1-8b-Instruct using the cache

```sh
oc create -n nim -f configs/software/nim/meta-service.yaml
```

Wait for the NIM service to be ready

```sh
oc wait -n nim --for=jsonpath='{.spec.status}'='Ready' nimservice meta-llama3-8b-instruct --timeout=300s
```

> TODO: Is there a way to view the containerfile of the NIM service that is deployed? What is under the hood?

Expose the service

```sh
oc expose -n nim svc meta-llama3-8b-instruct
```

Smoke test

```sh
NIM_META_URL=$(oc get route -n nim meta-llama3-8b-instruct --template='http://{{.spec.host}}'))

curl -X "POST" \
 $NIM_META_URL/v1/chat/completions' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "meta/llama3-8b-instruct",
        "messages": [
        {
          "content":"What should I do for a 4 day vacation at Cape Hatteras National Seashore?",
          "role": "user"
        }],
        "top_p": 1,
        "n": 1,
        "max_tokens": 1024,
        "stream": false,
        "frequency_penalty": 0.0,
        "stop": ["STOP"]
      }'
```

> TODO: Link to time slice GPU to create two replicas

Update the NIM service with autoscaling

```sh
oc apply -n nim -f configs/software/nim/meta-service-hpa.yaml
```

Smoke test

```sh
while true; do sleep 1; curl -X "POST" \
 $NIM_META_URL/v1/chat/completions' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "meta/llama3-8b-instruct",
        "messages": [
        {
          "content":"What should I do for a 4 day vacation at Cape Hatteras National Seashore?",
          "role": "user"
        }],
        "top_p": 1,
        "n": 1,
        "max_tokens": 1024,
        "stream": false,
        "frequency_penalty": 0.0, 
        "stop": ["STOP"]
      }'; echo "";done
```

Check that NIM service autoscaled

```sh
oc -n nim nimservice meta-llama3-8b-instruct
```

Delete NIM service

```sh
oc -n nim delete nimservice meta-llama3-8b-instruct
```

## NIM Pipelines

> TODO

## NeMo

> TODO: Bring a NeMo custom model from the NeMo section and deploy it as a NIM
