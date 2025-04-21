# Kubernetes AI (KAI) Scheduler

> [!IMPORTANT]
> Make sure you have configured Time Slicing, see [Time Slicing](docs/infra/gpu-timeslice.md).

## Setup

Install the scheduler:

```bash
helm upgrade -i kai-scheduler oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler -n kai-scheduler --create-namespace --version v0.4.3
```

Create a sandbox project:

```bash
oc new-project sandbox
```

Add scc role:

```bash
oc adm policy add-scc-to-user anyuid -z default
```

## Hierarchical Queues
 
Create queues:

```bash
oc create -f https://raw.githubusercontent.com/NVIDIA/KAI-Scheduler/refs/heads/main/docs/quickstart/queues.yaml
```

## Batch Scheduling

Batch Scheduling includes gang scheduling, which means all pods are scheduled together at once, or none are scheduled until resources for all become available.

### Kubeflow Training Operator

We need to install the Kubeflow Training Operator because the Nvidia gang scheduling example uses `PyTorchJob` from the Kubeflow Training Operator.

First, install the [OpenShift AI Operator](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.19/html/installing_and_uninstalling_openshift_ai_self-managed/installing-and-deploying-openshift-ai_install#installing-the-openshift-data-science-operator_operator-install).

Second, configure a [Data Science Cluster](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.19/html/installing_and_uninstalling_openshift_ai_self-managed/installing-and-deploying-openshift-ai_install#installing-openshift-ai-components-using-cli_component-install) and set the following:

```text
trainingoperator:
  managementState: Managed
```

Everything else can be set to `Removed`.

### Example

Run a distributed job with gang scheduling:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/batch/pytorch-job.yaml
```

Wait for the job (this can take ~5 minutes):

```bash
oc wait pytorchjob/pytorch-dist-mnist-nccl --for=condition=Running --timeout=600s -n sandbox
```

View the pods and see that all are scheduled at once:

```bash
oc get pods -l training.kubeflow.org/job-name=pytorch-dist-mnist-nccl
```

Delete the job:

```bash
oc delete pytorchjob pytorch-dist-mnist-nccl
```

### Workload Priority

Modify the `test` queue with a GPU limit of `1`:

```bash
oc apply -f https://raw.githubusercontent.com/NVIDIA/KAI-Scheduler/refs/heads/main/docs/priority/example/limited-queue.yaml
```

Submit a pod with `train` priority (value 50) to the queue:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/example/train-priority-pod.yaml
```

View the pod:

```bash
oc get pods -l runai/queue=test
```

```text
NAME        READY   STATUS    RESTARTS   AGE
train-pod   1/1     Running   0          
```

Submit a pod with 'build` priority (value 100) to the queue:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/example/build-priority-pod.yaml
```

The `build` pod will preempt the `train` pod.

View the pods:

```bash
oc get pods -l runai/queue=test
```

```text
NAME        READY   STATUS        RESTARTS   AGE
build-pod   1/1     Running       0         
train-pod   1/1     Terminating   0          
```

### Cleanup

Delete the pods:

```bash
oc delete pods --all -n sandbox
```

Restore the queue limit:

```bash
oc apply -f https://raw.githubusercontent.com/NVIDIA/KAI-Scheduler/refs/heads/main/docs/quickstart/queues.yaml
```

### GPU Sharing

KAI supports GPU sharing by requesting a portion of GPU memory, but this is not enforced by the scheduler. Per the [docs](https://github.com/NVIDIA/KAI-Scheduler/blob/main/docs/gpu-sharing/README.md), *KAI Scheduler does not enforce memory allocation limit or performs memory isolation between processes. In order to make sure the pods share the GPU device nicely it is important that the running processes will allocate GPU memory up to the requested amount and not beyond that. *

Enable GPU sharing:

```bash
helm upgrade -i kai-scheduler oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler -n kai-scheduler --create-namespace --version v0.4.3 --set "global.gpuSharing=true"
```

Submit a pod that requests `0.5` of GPU memory:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/gpu-sharing/gpu-sharing.yaml
```

View the scheduled pod:

```bash
oc get pods -l runai/queue=test -n sandbox
```

```text
NAME          READY   STATUS    RESTARTS   AGE
gpu-sharing   1/1     Running   0          
```

It is **really important** to note that requested GPU memory does not decrement the allocatable capacity on the node:

```bash
oc get node --selector=nvidia.com/gpu.product=NVIDIA-L4-SHARED -o jsonpath-as-json='{.items[0].status.allocatable}'
```

> Node still shows full GPU capacity is available

```text
[
    {
        "cpu": "3500m",
        "ephemeral-storage": "95018478229",
        "hugepages-1Gi": "0",
        "hugepages-2Mi": "0",
        "memory": "14589080Ki",
        "nvidia.com/gpu": "4",
        "pods": "250"
    }
]
```

Furthermore, it is also **extremely important** to note that the process on the GPU has full access to the GPU memory (even though we asked for `0.5` GPU memory).

```bash
> todo
```


