# Riva

## Setup

> [!IMPORTANT]
> Make sure you have configured [GPU Operator](../infra/gpu-operator.md).\
> Make sure you have your Nvidia API key, see the [Prerequisites](../prereqs.md).

Download helm chart

```bash
helm fetch https://helm.ngc.nvidia.com/nvidia/riva/charts/riva-api-2.19.0.tgz \
        --username=\$oauthtoken --password=$NGC_API_KEY --untar
```

Edit the `values.yaml` with

1. `ngcCredentials` put in your `NGC_API_KEY` and `email`
1. `persistentVolumeClaim` change `usePVC` to `true` and set `storageClassName` (e.g. `gp3-csi` in AWS) and set `storageAccessMode` to `ReadWriteOnce`
1. Append the following models under `ngcModelConfigs.triton0.models`:

> Note: Uncomment the model you want loaded into Riva. In the example below, we are loading the [Canary 1B](https://build.nvidia.com/nvidia/canary-1b-asr) model:

```text
      - nvidia/riva/rmir_asr_canary_1b_ofl:2.19.0
      # - nvidia/riva/rmir_asr_canary_0-6b_turbo_ofl:2.19.0
      # - nvidia/riva/rmir_asr_whisper_large_ofl:2.19.0
```

Create project:

```bash
oc new-project riva
```

Assign RBAC permissions to service account for Riva:

```bash
oc adm policy add-scc-to-user nonroot-v2 -z default
```

> Add a toleration for GPUs to the `riva-api/templates/triton.yaml` file

Deploy:

```bash
helm install riva-api riva-api
```

Fix secret:

> There is a bug in the helm chart and requires to manually create the model pull secret

```bash
oc create secret generic modelpullsecret --from-literal=apikey=$NGC_API_KEY
```

## Speech to Text API

Riva exposes a gRPC API instead of HTTP, so it needs a client

```bash
oc create -f configs/software/riva/client.yaml
```

Get a reference to the client pod

```bash
RIVA_CLIENT=$(oc get pods -l app=rivaasrclient -o jsonpath='{.items[0].metadata.name}')
```

Run a transcription smoke test

```bash
oc exec $RIVA_CLIENT -- clients/riva_streaming_asr_client --print_transcripts \
  --audio_file=/opt/riva/wav/en-US_sample.wav --automatic_punctuation=true --riva_uri=riva-api:50051
```

Run a streaming transcription

```bash
oc exec $RIVA_CLIENT -- python3 examples/transcribe_file.py \
  --input-file /opt/riva/wav/en-US_sample.wav --server riva-api:50051
```

Run an offline transcription

```bash
oc exec $RIVA_CLIENT -- python3 examples/transcribe_file_offline.py \
  --input-file /opt/riva/wav/en-US_sample.wav --server riva-api:50051
```

List available ASR models in your Riva server

```bash
oc exec $RIVA_CLIENT -- python3 examples/transcribe_file.py --list-models --server riva-api:50051
```

#### Model - Canary

> Canary only offers offline transcription in Riva

```bash
oc exec $RIVA_CLIENT -- python3 examples/transcribe_file_offline.py --model-name canary-1b-multi-asr-offline-asr-bls-ensemble\
  --input-file /opt/riva/wav/en-US_sample.wav --server riva-api:50051
```

#### Model - Conformer

Run a streaming transcription with the conformer streaming model

```bash
oc exec $RIVA_CLIENT -- python3 examples/transcribe_file.py --model-name conformer-en-US-asr-streaming-asr-bls-ensemble\
  --input-file /opt/riva/wav/en-US_sample.wav --server riva-api:50051
```

#### Model - Parakeet

Run a streaming transcription with the parakeet streaming model

```bash
oc exec $RIVA_CLIENT -- python3 examples/transcribe_file.py --model-name parakeet-0.6b-en-US-asr-streaming-throughput-asr-bls-ensemble\
  --input-file /opt/riva/wav/en-US_sample.wav --server riva-api:50051
```

## Audio Mic Transcription

**Optional: Transcribe live with an audio mic**

> Note: This requires you to download the repo to your machine that has an audio mic

In one terminal,

```sh
oc port-forward service/riva-api 8443:riva-speech
```

In another terminal:

Clone the repo

```sh
git clone git@github.com:nvidia-riva/python-clients.git
cd python-clients
```

Install Python Audio

```sh
pipenv install pyaudio
```

Install dependencies

```sh
pipenv install -r requirements.txt
```

Install Riva Client

```sh
pipenv install nvidia-riva-client
```

Activate shell

```sh
pipenv shell
```

Run Audio mic transcription

```sh
python3 scripts/asr/transcribe_mic.py --server localhost:8443
```

