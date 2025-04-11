# GPU Operator

> TODO: Add preamble

> TODO: Add note about [support](https://access.redhat.com/solutions/5174941)

## Create GPU Machines

Let's add machines with GPUs to your cluster. You will create a [g6 instance](https://aws.amazon.com/ec2/instance-types/g6/) which feature Nvidia L4 GPUs.

View the MachineSets in your cluster:

```bash
oc get machinesets -n openshift-machine-api
```

```text
NAME                                    DESIRED   CURRENT   READY   AVAILABLE   AGE
cluster-xxxxx-xxxxx-worker-us-xxxx-xx   1         1         1       1               
```

Make a copy of the existing MachineSet configuration:

```bash
MACHINESET=$(oc get machineset -n openshift-machine-api -o jsonpath='{.items[0].metadata.name}')
oc get machineset $MACHINESET -n openshift-machine-api -o yaml > scratch/gpu-machineset.yaml
```

Edit `scratch/gpu-machineset.yaml`

  - [ ] Delete `creationTimestamp`, `generation`, `resourceVersion`, `uid`
  - [ ] Set `.metadata.name` to `gpu-machineset`
  - [ ] Set `.spec.replicas` to `1`
  - [ ] Set `.spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"]` to `gpu-machineset`
  - [ ] Set `.spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"]` to `gpu-machineset`
  - [ ] Set `.spec.template.spec.providerSpec.value.instanceType` to `g6.4xlarge`
 
Finally, further edit `scratch/gpu-machineset.yaml` to add a taint, so that non-GPU specific workloads do not run on those machines.

The Nvidia GPU Operator by default adds a toleration for the key `nvidia.com/gpu`, so let's use this taint key.

The taint should look like:

```text
apiVersion: machine.openshift.io/v1beta1
kind: MachineSet
metadata:
spec:
  [...]
  template:
    [...]
    spec:
      [...]
      taints:
      - effect: NoSchedule        
        key: nvidia.com/gpu
        value: ''
```

Create your GPU MachineSet:

```bash
oc create -f scratch/gpu-machineset.yaml
```     

Verify the GPU MachineSet:

```bash
oc get machinesets -n openshift-machine-api
```

```text
NAME                                    DESIRED   CURRENT   READY   AVAILABLE   AGE
cluster-xxxxx-xxxxx-worker-us-xxxx-xx   1         1         1       1               
gpu-machineset                          1         1                               
```

## Install Node Feature Discovery Operator

> TODO: Add preamble

> Not strictly required for GPU Operator to work, but necessary to obtain support from Nvidia

The NFD Operator uses vendor PCI IDs to identify hardware in a node.

Below are some of the [PCI vendor ID assignments](https://pcisig.com/membership/member-companies?combine=10de):

| PCI id | Vendor |
| ------ | ------ |
| `10de` | NVIDIA |
| `1d0f` | AWS    |
| `1002` | AMD    |
| `8086` | Intel  |

Let's install the NFD Operator. First list the available operators for installation searching for Node Feature Discovery (NFD)

```bash
oc get packagemanifests -n openshift-marketplace | grep nfd
```

```text
openshift-nfd-operator                           Community Operators   6h43m
nfd                                              Red Hat Operators     6h43m
```

Create the NFD Operator:

```bash
oc create -f configs/infra/gpu/nfd-operator.yaml
```

Use the `rollout` command to verify the deployment. You might get an error if the deployment does not exist yet. In that case, wait a few seconds and try again.

Wait for the operator deployment:

```bash
oc rollout status deploy/nfd-controller-manager -n openshift-nfd --timeout=300s     
```

```text
deployment "nfd-controller-manager" successfully rolled out
```

Now that the operator is deployed, we need to create a `NodeFeatureDiscovery` custom resource which will deploy DaemonSets that discover hardware features and label the nodes accordingly.

Create the `NodeFeatureDiscovery` instance:

```bash
oc create -f configs/infra/gpu/nfd-instance.yaml
```

Verify the NFD pods are `Running` on the cluster nodes polling for devices:

```bash
oc get pods -n openshift-nfd
```

```
NAME                                      READY   STATUS    RESTARTS   AGE
nfd-controller-manager-xxxxxxxxxx-xxxxx   2/2     Running   0            
nfd-master-xxxxxxxxxx-xxxxx               1/1     Running   0            
nfd-worker-xxxxx                          1/1     Running   0            
nfd-worker-xxxxx                          1/1     Running   0            
nfd-worker-xxxxx                          1/1     Running   0            
```

Verify the GPU device (NVIDIA uses the PCI ID `10de`) is discovered on the GPU node. This means the NFD Operator correctly identified the node from the GPU-enabled MachineSet.

```bash
oc describe node | egrep 'Roles|pci' | grep -v master
```

```
Roles:              worker
                    feature.node.kubernetes.io/pci-10de.present=true
                    feature.node.kubernetes.io/pci-1d0f.present=true
                    feature.node.kubernetes.io/pci-1d0f.present=true
Roles:              worker
                    feature.node.kubernetes.io/pci-1d0f.present=true
```

## Install the NVIDIA GPU Operator

> TODO: Add preamble

List the available operators for installation searching for GPU:

```bash
oc get packagemanifests -n openshift-marketplace | grep gpu
```

```text
amd-gpu-operator                                   Community Operators   8h
amd-gpu-operator                                   Certified Operators   8h
gpu-operator-certified                             Certified Operators   8h
```

Create the Nvidia GPU Operator:

```bash
oc create -f configs/infra/gpu/nvidia-gpu-operator.yaml
```

Use the `rollout` command to verify the deployment. You might get an error if the deployment does not exist yet. In that case, wait a few seconds and try again.

Wait for the operator deployment:

```bash
oc rollout status deploy/gpu-operator -n nvidia-gpu-operator --timeout=300s
```

```text
deployment "gpu-operator" successfully rolled out`
```

Verify the version of the GPU operator that was installed:

```bash
oc get ip -n nvidia-gpu-operator
```

```
NAME            CSV                              APPROVAL    APPROVED
install-xxxxx   gpu-operator-certified.v24.9.2   Automatic   true
```

> [!NOTE]
> The CSV version should match the latest supported [version](https://docs.nvidia.com/ai-enterprise/release-6/latest/support/support-matrix.html#supported-nvidia-configs/infrastructure-software) of the GPU Operator.

> TODO: Explain Cluster Policy

Create a Cluster Policy configuration:

```bash
oc get csv -n nvidia-gpu-operator -l operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator \
  -ojsonpath='{.items[0].metadata.annotations.alm-examples}' | jq '.[0]' > scratch/nvidia-gpu-clusterpolicy.json
```

> [!IMPORTANT]
> If you decided to use a custom taint key instead of `nvidia.com/gpu`, then you will need to modify the cluster policy file and add a toleration for your custom key.
> See this [example](https://github.com/NVIDIA/gpu-operator/blob/main/deployments/gpu-operator/values.yaml#L39) for where to set this in the Cluster Policy.

Apply the configuration:

```bash
oc apply -n nvidia-gpu-operator -f scratch/nvidia-gpu-clusterpolicy.json
```

Wait for the GPU Operator components to finish installing. This can take up to 20 minutes to complete:

```bash
oc wait clusterpolicy/gpu-cluster-policy --for=condition=Ready --timeout=600s -n nvidia-gpu-operator
```

```text
clusterpolicy.nvidia.com/gpu-cluster-policy condition met
```

Verify the successful installation of the NVIDIA driver:

```bash
oc get pod -l openshift.driver-toolkit -n nvidia-gpu-operator
```

```text
NAME                                                  READY   STATUS    RESTARTS   AGE
nvidia-driver-daemonset-417.94.202503060903-0-xxxxx   2/2     Running   0            
```

## Smoke Test

Use the [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/) program to test GPU access.

```bash
oc exec -n nvidia-gpu-operator $(oc get pod -n nvidia-gpu-operator -l openshift.driver-toolkit -ojsonpath='{.items[0].metadata.name}') -- nvidia-smi
```

You should see the `NVIDIA L4` GPU you provisioned.

Run CUDA VectorAdd:

```bash
oc create -f configs/infra/gpu/nvidia-gpu-sample-app.yaml -n sandbox
```

Check logs

```bash
oc logs cuda-vectoradd
```

Next: [Time Slice](gpu-timeslice.md)
