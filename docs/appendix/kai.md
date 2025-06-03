# Kubernetes AI (KAI) Scheduler

> [!IMPORTANT]
> Configure [GPU Operator](../infra/gpu-operator.md).\
> Configure [Time Slicing](../infra/gpu-timeslice.md).

## Setup

Install the scheduler:

```bash
helm upgrade -i kai-scheduler oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace --version v0.4.3
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

#### Single Queue

Let's demonstrate how to preempt a pod in a single queue. This functionality is identical to default Kubernetes priority/preemption.

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

#### Two Queues

In this example, we demonstrate how to preempt pods in two different queues, which is a differentiator compared to default Kubernetes priority/preemption.

Delete the `test` queue:

```bash
oc delete queue test
```

Create two queues `team-a` and `team-b`:

```bash
oc apply -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/two-team/two-team-queues.yaml 
```

Submit a pod with `train` priority (value 50) to `team-a` and `team-b` queues:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/two-team/train-team-a.yaml
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/two-team/train-team-b.yaml
```

```bash
oc get pods -l 'runai/queue in (team-a,team-b)'
```

```text
NAME               READY   STATUS    RESTARTS   AGE
train-pod-team-a   1/1     Running   0          
train-pod-team-b   1/1     Running   0          
```

Preempt both of these pods with `build` priority (value 100) in each queue:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/two-team/build-team-a.yaml
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/two-team/build-team-b.yaml
```

View the pods:

```bash
oc get pods -l 'runai/queue in (team-a,team-b)'
```

```text
NAME               READY   STATUS        RESTARTS   AGE
build-pod-team-a   1/1     Running       0          
build-pod-team-b   1/1     Running       0          
train-pod-team-a   1/1     Terminating   0          
train-pod-team-b   1/1     Terminating   0          
```

#### Three Queues

Let's run a more complicated scenario. In this example, there are three teams (Team A,B,C) each with their own queue. Each team has a quota of `1` GPU with a limit of `3`. This means that each team is guaranteed `1` GPU and can use up to `3` GPUs if the other teams are not using their GPUs.

However, once teams start to use their "faire share" of GPUs, the team that is using "more GPUs" should have their workload preempted.

Delete the pods:

```bash
oc delete pods --all -n sandbox
```

Configure three queues:

```bash
oc apply -f https://github.com/theckang/KAI-Scheduler/blob/main/docs/priority/team-example/three-team/three-team-queues.yaml 
```

Create a workload that uses 3 GPUs in `team-a` queue:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/three-team/train-team-a.yaml
```

View the pods

```bash
oc get pods -l 'runai/queue in (team-a,team-b,team-c)'
```

```text
NAME               READY   STATUS    RESTARTS   AGE
train-pod-team-a   1/1     Running   0          
```

Create a workload that uses 1 GPU in `team-b` queue:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/three-team/train-team-b.yaml 
```

This workload fits because the parent queue `default` has no limit and can use up to our hard limit (4 GPUs from time slicing).

View the pods

```bash
oc get pods -l 'runai/queue in (team-a,team-b,team-c)'
```

```text
NAME               READY   STATUS    RESTARTS   AGE
train-pod-team-a   1/1     Running   0          
train-pod-team-b   1/1     Running   0          
```

Create a workload that uses 1 GPU in `team-c` queue:

```bash
oc create -f https://raw.githubusercontent.com/theckang/KAI-Scheduler/refs/heads/main/docs/priority/team-example/three-team/train-team-c.yaml
```

This workload does not fit since 4 GPUs have been allocated. Team A is using 3 GPUs (over its quota of `1`) so the workload in `team-a` is preempted:

View the pods:

```bash
oc get pods -l 'runai/queue in (team-a,team-b,team-c)'
```

```text
NAME               READY   STATUS        RESTARTS   AGE
train-pod-team-a   1/1     Terminating   0          
train-pod-team-b   1/1     Running       0          
train-pod-team-c   0/1     Pending       0          
```

Team A's workload is preempted so that Team C can get their fair share of its GPU back.

View the pods after some time:

```bash
oc get pods -l 'runai/queue in (team-a,team-b,team-c)'
```

```text
NAME               READY   STATUS    RESTARTS   AGE
train-pod-team-b   1/1     Running   0          
train-pod-team-c   1/1     Running   0          
```

Team B and Team C are now running their workloads on GPUs.

### Cleanup

Delete the pods:

```bash
oc delete pods --all -n sandbox
```

Delete any queues:

```bash
oc delete queues --all
```

Restore the default/test queues and limit:

```bash
oc apply -f https://raw.githubusercontent.com/NVIDIA/KAI-Scheduler/refs/heads/main/docs/quickstart/queues.yaml
```

### GPU Sharing

KAI supports GPU sharing by requesting a portion of GPU memory, but this is not enforced by the scheduler. Per the [docs](https://github.com/NVIDIA/KAI-Scheduler/blob/main/docs/gpu-sharing/README.md), *KAI Scheduler does not enforce memory allocation limit or performs memory isolation between processes. In order to make sure the pods share the GPU device nicely it is important that the running processes will allocate GPU memory up to the requested amount and not beyond that.*

Enable GPU sharing:

```bash
helm upgrade -i kai-scheduler oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler -n kai-scheduler --create-namespace --version v0.4.3 --set "global.gpuSharing=true"
```

View a sample deployment with `gpu-fraction` set:

```bash
cat configs/appendix/whisper-gpu-sharing.yaml | grep annotations -A 1
```

```text

      annotations:
        gpu-fraction: "0.5"

```

Submit the deployment that requests `0.5` of GPU memory:

```bash
oc create -f configs/appendix/whisper-gpu-sharing.yaml
```

Wait for the deployment:

```bash
oc rollout status deploy/whisper-tiny -n sandbox --timeout=600s
```

It is **really important** to note that the requested GPU memory does not decrement the allocatable capacity on the node:

```bash
oc get node --selector=nvidia.com/gpu.product=NVIDIA-A10G-SHARED -o jsonpath-as-json='{.items[0].status.allocatable}'
```

> The node shows full GPU capacity is available even though we deployed a pod that requested half of the GPU's memory

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
oc exec $(oc get pod -n sandbox -l runai/queue=test -ojsonpath='{.items[0].metadata.name}') -- nvidia-smi
```

```text
---
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:31:00.0 Off |                    0 |
| N/A   38C    P0             26W /   72W |   17143MiB /  23034MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
---
```

The NVIDIA L4 has 24GB of memory and the deployment currently uses ~75% of the GPU's total memory, even though we asked the scheduler to request `0.5` of GPU memory.

### Cleanup

Delete the deployment:

```bash
oc delete deploy/whisper-tiny -n sandbox
```
