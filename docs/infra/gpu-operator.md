# GPU Operator

> TODO: Add preamble

> TODO: Add note about [support](https://access.redhat.com/solutions/5174941)

## Create GPU Nodes

> [!NOTE]
> To demonstrate NIM, a [g6e.xlarge](https://aws.amazon.com/ec2/instance-types/g6e/) will be provisioned.

> [!IMPORTANT]
> To demonstrate MIG, a [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/) will be provisioned.
> This VM has 8 x A100 GPUs and AWS does not offer a smaller VM size with these GPUs.
> Please be sure to delete the MIG machines once you have completed the instructions to save on cost! 

View the MachineSets in your cluster:

```sh
oc get machinesets -n openshift-machine-api
```

```text
NAME                                    DESIRED   CURRENT   READY   AVAILABLE   AGE
cluster-xxxxx-xxxxx-worker-us-xxxx-xx   0         0                                
```

Make a copy of the existing MachineSet:

```sh 
MACHINESET=$(oc get machineset -n openshift-machine-api -o jsonpath='{.items[0].metadata.name}')
oc get machineset $MACHINESET -n openshift-machine-api -o yaml > scratch/gpu-machineset.yaml
```

Edit `scratch/gpu-machineset.yaml`

  - [ ] Delete `creationTimestamp`, `generation`, `resourceVersion`, `uid`
  - [ ] Set `.metadata.name` to `gpu-machineset`
  - [ ] Set `.spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"]` to `gpu-machineset`
  - [ ] Set `.spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"]` to `gpu-machineset`
  - [ ] Set `.spec.replicas` to `1`
  - [ ] Set `.spec.template.spec.providerSpec.value.instanceType` to `g6e.xlarge`

Create your new MachineSet:

```sh
oc create -f scratch/gpu-machineset.yaml
```      

Verify the new MachineSet:

```sh
oc get machinesets -n openshift-machine-api
```

```text
NAME                                    DESIRED   CURRENT   READY   AVAILABLE   AGE
cluster-xxxxx-xxxxx-worker-us-xxxx-xx   0         0                                
gpu-machineset                          1         1                                
```

## Install Node Feature Discovery Operator

> TODO: Add preamble

> Not strictly required for GPU Operator to work, but necessary to obtain support from Nvidia

List the available operators for installation searching for Node Feature Discovery (NFD)

```sh
oc get packagemanifests -n openshift-marketplace | grep nfd
```

```text
openshift-nfd-operator                           Community Operators   6h43m
nfd                                              Red Hat Operators     6h43m
```

Create the NFD Operator:

```sh
oc create -f configs/infra/gpu/nfd-operator.yaml
```

Verify Operator is installed and running:

> [!TIP]
> You might get an error if the deployment does not exist yet. Wait a few seconds and try again.

```sh
oc rollout status deploy/nfd-controller-manager -n openshift-nfd --timeout=300s      
```

```text
Waiting for deployment "nfd-controller-manager" rollout to finish: 0 of 1 updated replicas are available...
deployment "nfd-controller-manager" successfully rolled out
```

> [!NOTE]
> After installing the NFD Operator, you create instance that installs the `nfd-master` and one `nfd-worker` pod for each compute node. [More Info](https://docs.openshift.com/container-platform/4.15/hardware_enablement/psap-node-feature-discovery-operator.html#Configure-node-feature-discovery-operator-sources_psap-node-feature-discovery-operator)

Create the nfd instance object:

```sh
oc create -f configs/infra/gpu/nfd-instance.yaml
```

> [!IMPORTANT]
> The NFD Operator uses vendor PCI IDs to identify hardware in a node.

Below are some of the [PCI vendor ID assignments](https://pcisig.com/membership/member-companies?combine=10de):

| PCI id | Vendor |
| ------ | ------ |
| `10de` | NVIDIA |
| `1d0f` | AWS    |
| `1002` | AMD    |
| `8086` | Intel  |

Verify the NFD pods are `Running` on the cluster nodes polling for devices

```sh
oc get pods -n openshift-nfd
```

Expected output

```
NAME                                      READY   STATUS    RESTARTS   AGE
nfd-controller-manager-xxxxxxxxxx-xxxxx   2/2     Running   0             
nfd-master-xxxxxxxxxx-xxxxx               1/1     Running   0             
nfd-worker-xxxxx                          1/1     Running   0             
nfd-worker-xxxxx                          1/1     Running   0             
nfd-worker-xxxxx                          1/1     Running   0             
```

Verify the GPU device (NVIDIA uses the PCI ID `10de`) is discovered on the GPU node. This means the NFD Operator correctly identified the node from the GPU-enabled MachineSet.

```sh
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

```sh
oc get packagemanifests -n openshift-marketplace | grep gpu
```

```text
amd-gpu-operator                                   Community Operators   8h
amd-gpu-operator                                   Certified Operators   8h
gpu-operator-certified                             Certified Operators   8h
```

Create the Nvidia GPU Operator:

```sh
oc create -f configs/infra/gpu/nvidia-gpu-operator.yaml
```

Wait for Operator to finish installing:

```sh
oc rollout status deploy/gpu-operator -n nvidia-gpu-operator --timeout=300s
```

```
deployment "gpu-operator" successfully rolled out`
```

Verify the version of the GPU operator that was installed:

```sh
oc get ip -n nvidia-gpu-operator
```

```
NAME            CSV                              APPROVAL    APPROVED
install-xxxxx   gpu-operator-certified.v24.9.2   Automatic   true
```

> [!IMPORTANT]
> The CSV version should match the latest supported [version](https://docs.nvidia.com/ai-enterprise/release-6/latest/support/support-matrix.html#supported-nvidia-configs/infrastructure-software) of the GPU Operator.
 
Create the cluster policy:

```sh
oc get csv -n nvidia-gpu-operator -l operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator \
  -ojsonpath='{.items[0].metadata.annotations.alm-examples}' | jq '.[0]' > scratch/nvidia-gpu-clusterpolicy.json
```

Apply the clusterpolicy:

```sh
oc apply -n nvidia-gpu-operator -f scratch/nvidia-gpu-clusterpolicy.json
```

Wait for the GPU Operator components to finish installing:

> [!IMPORTANT]
> This step can take up to 20 minutes to complete!

```sh
oc wait clusterpolicy/gpu-cluster-policy --for=condition=Ready --timeout=600s -n nvidia-gpu-operator
```

Verify the successful installation of the NVIDIA driver:

```sh
oc get pod -l openshift.driver-toolkit -n nvidia-gpu-operator
```

```text
NAME                                                  READY   STATUS    RESTARTS   AGE
nvidia-driver-daemonset-417.94.202503060903-0-xxxxx   2/2     Running   0             
```

## Smoke Test

Test GPU Access

> [!NOTE]
> Nvidia System Management Interface `nvidia-smi` shows memory usage, GPU utilization, and the temperature of the GPU.

```sh
oc exec -n nvidia-gpu-operator $(oc get pod -n nvidia-gpu-operator -l openshift.driver-toolkit -ojsonpath='{.items[0].metadata.name}') -- nvidia-smi
```

Run CUDA VectorAdd

```sh
oc create -f configs/infra/gpu/nvidia-gpu-sample-app.yaml -n sandbox
```

Check logs

```sh
oc logs cuda-vectoradd
```
