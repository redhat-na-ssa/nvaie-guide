# Multi-Instance GPU

> [!IMPORTANT]
> Configure [GPU Operator](./gpu-operator.md).

Another way to share a GPU is Multi-Instance GPU ([MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/)). MIG can partition a GPU up to seven instances starting with the Ampere generation or later. Unlike time slicing, each GPU instance has dedicated hardware and fault isolation.

## GPU MIG Machines

> [!IMPORTANT]
> To demonstrate MIG, a [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/) will be provisioned.
> This VM has 8 x A100 GPUs and AWS does not offer a smaller VM size with these GPUs.
> Please be sure to delete the MIG machines once you are done to save on cost!

Follow the instructions for the [GPU Operator](./gpu-operator.md) but instead of creating a [g5 instance](https://aws.amazon.com/ec2/instance-types/g5/), let's create a [p4d instance](https://aws.amazon.com/ec2/instance-types/p4/). Specifically:

Edit `scratch/gpu-machineset.yaml`

  - [ ] Set `.spec.template.spec.providerSpec.value.instanceType` to `p4d.24xlarge`


After the machine is created, set a reference to the MIG enabled node:

```bash
NODE_NAME=$(oc get node --selector=node.kubernetes.io/instance-type=p4d.24xlarge -o jsonpath='{.items[0].metadata.name}')
```

## Single Strategy

In this strategy, each MIG instance has the same compute/memory size and the MIG instances are exposed as identical `nvidia.com/gpu` resources.

The `p4d.24xlarge` instance has 8 GPUs each with 40G of memory. You can view the available single geometries for this GPU [here](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/mig-ocp.html#id2). 

First, define the `single` strategy in the `ClusterPolicy` custom resource:

```bash
oc patch clusterpolicy gpu-cluster-policy \
    -n nvidia-gpu-operator --type json \
    -p '[{"op": "replace", "path": "/spec/mig/strategy", "value": single}]'
```

Set the desired MIG geometry. In this example, we will pick the `all-3g.20gb` configuration: 

```bash
MIG_CONFIGURATION=all-3g.20gb
```

Apply the MIG geometry to your existing GPU machine:

```bash
oc label --overwrite node $NODE_NAME \
    nvidia.com/mig.config=$MIG_CONFIGURATION --overwrite
```

View the partitioned MIG resources:

```bash
oc get node $NODE_NAME -o jsonpath-as-json='{.status.allocatable}'
```

```text
[
    {
        ...
        "nvidia.com/gpu: "16",
        ...
    }
]
```

## Mixed Strategy

In this strategy, you can mix different compute/memory sizes for partitioned MIG instances. Each partitioned MIG instance is exposed as a resource with its full partition size in its name, e.g. `nvidia.com/mig-1g.5gb`.

The `p4d.24xlarge` instance has 8 GPUs each with 40G of memory. You can view an example of a mixed geometry for this GPU [here](https://docs.nvidia.com/datacenter/cloud-native/openshift/latest/mig-ocp.html#id3).

First, define the strategy in the `ClusterPolicy` custom resource:

```bash
oc patch clusterpolicy gpu-cluster-policy \
    -n nvidia-gpu-operator --type json \
    -p '[{"op": "replace", "path": "/spec/mig/strategy", "value": mixed}]'
```

Set the desired MIG geometry. In this example, we will pick the `all-balanced` configuration:

```bash
MIG_CONFIGURATION=all-balanced
```

Apply the MIG geometry to your existing GPU machine:

```bash
oc label --overwrite node $NODE_NAME \
    nvidia.com/mig.config=$MIG_CONFIGURATION --overwrite
```

View the exposed MIG instances:

```bash
oc get node $NODE_NAME -o jsonpath-as-json='{.status.allocatable}'
```

```text
[
    {
        ...
        "nvidia.com/mig-1g.5gb": "16",
        "nvidia.com/mig-2g.10gb": "8",
        "nvidia.com/mig-3g.20gb": "8",
        ...        
    }
]
```

## Cleanup

Disable MIG:

```bash
MIG_CONFIGURATION=all-disabled && \
  oc label --overwrite node $NODE_NAME \
    nvidia.com/mig.config=$MIG_CONFIGURATION --overwrite
```

Delete the GPU machine set:

```bash
oc delete machineset gpu-machineset -n openshift-machine-api
```
