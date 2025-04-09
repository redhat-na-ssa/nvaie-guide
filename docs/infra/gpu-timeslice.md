# Time Slice

By default, only one workload can run on a single GPU. In some use cases though, you may want to share (or partition) the GPU to allow multiple workloads to run on the single GPU. 

For example, you might want to run multiple copies of an inference service on a GPU to increase the throughput of requests. Note that this comes at the cost of higher latency per request, because multiple inference services are now sharing the same GPU.

One way to share a GPU is time slicing. While time slicing allows you to multiplex N workloads to a GPU, there is no memory or fault isolation for workloads running on that GPU.

First, we need to define the sharing configuration. Here is the configuration you will use:

```text
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: device-plugin-config
  namespace: nvidia-gpu-operator
data:
  NVIDIA-L4: |-
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4
```

In this configuration, we will time slice the GPU into 4 partitions.

Create this configuration:

```bash
oc create -f configs/infra/timeslice/device-plugin-config.yaml
```

Add this configuration to the `ClusterPolicy` custom resource:

```bash
oc patch clusterpolicy gpu-cluster-policy \
    -n nvidia-gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "device-plugin-config"}}}}'
```

The GPU Operator is now aware of the sharing configuration. Now you have to apply the configuration to the specific GPU machines that you want partitioned. 

In this case, we only want to apply this configuration to the GPU MachineSet that you created.

Apply the configuration to your existing GPU machine:

```bash
oc label --overwrite node \
    --selector=nvidia.com/gpu.product=NVIDIA-L4 \
    nvidia.com/device-plugin.config=NVIDIA-L4
```

This will append a suffix `-SHARED` to the product label name.

Apply the configuration to your MachineSet so future GPU machines also get this configuration:

```bash
oc patch machineset gpu-machineset \
    -n openshift-machine-api --type merge \
    --patch '{"spec": {"template": {"spec": {"metadata": {"labels": {"nvidia.com/device-plugin.config": "NVIDIA-L4"}}}}}}'
```

Verify that time slicing was configured:

```bash
oc get node --selector=nvidia.com/gpu.product=NVIDIA-L4-SHARED -o jsonpath-as-json='{.items[0].status.capacity}'
```

```text
[
    {
        "cpu": "16",
        "ephemeral-storage": "104266732Ki",
        "hugepages-1Gi": "0",
        "hugepages-2Mi": "0",
        "memory": "63402072Ki",
        "nvidia.com/gpu": "4",
        "pods": "250"
    }
]
```

Next: [MIG](gpu-mig.md)
