# GPU Operator

> TODO: Add preamble\
> TODO: Add note about [support](https://access.redhat.com/solutions/5174941)

## Create GPU Nodes

> [!IMPORTANT]
> To demonstrate MIG, a [p4d.24xlarge](https://aws.amazon.com/ec2/instance-types/p4/) will be provisioned.
> This VM has 8 x A100 GPUs and AWS does not offer a smaller VM size with these GPUs.
> Please be sure to delete the MIG machines once you have completed the instructions to save on cost! 

## Steps

- [ ] View the MachineSet in the `openshift-machine-api` namespace

      oc get machinesets -n openshift-machine-api

> Expected output
>
> `NAME                                    DESIRED   CURRENT   READY   AVAILABLE   AGE`\
> `cluster-xxxxx-xxxxx-worker-us-xxxx-xx   0         0                                `

- [ ] Make two copies of the MachineSet and output the results to YAML files
 
      MACHINESET=$(oc get machineset -n openshift-machine-api -o jsonpath='{.items[0].metadata.name}')
      
      oc get machineset $MACHINESET -n openshift-machine-api -o yaml > scratch/timeslice-machineset.yaml

      oc get machineset $MACHINESET -n openshift-machine-api -o yaml > scratch/mig-machineset.yaml

- [ ] Edit `scratch/timeslice-machineset.yaml`

  - [ ] Delete `creationTimestamp`, `generation`, `resourceVersion`, `uid`
  - [ ] Set `.metadata.name` to `ts-machineset`
  - [ ] Set `.spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"]` to `ts-machineset`
  - [ ] Set `.spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"]` to `ts-machineset`
  - [ ] Set `.spec.replicas` to `1`
  - [ ] Set `.spec.template.spec.providerSpec.value.instanceType` to `g6.xlarge`

- [ ] Edit `scratch/mig-machineset.yaml`
  - [ ] Delete `creationTimestamp`, `generation`, `resourceVersion`, `uid`
  - [ ] Set `.metadata.name` to `mig-machineset`
  - [ ] Set `.spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"]` to `mig-machineset`
  - [ ] Set `.spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"]` to `mig-machineset`
  - [ ] Set `.spec.replicas` to `1`
  - [ ] Set `.spec.template.spec.providerSpec.value.instanceType` to `p4d.24xlarge`  

- [ ] Create MachineSets

      oc create -f scratch/timeslice-machineset.yaml
      
      oc create -f scratch/mig-machineset.yaml

- [ ] Verify the new MachineSets in the `openshift-machine-api` namespace

      oc get machinesets -n openshift-machine-api

> Expected output
>
> `NAME                                    DESIRED   CURRENT   READY   AVAILABLE   AGE`\
> `cluster-xxxxx-xxxxx-worker-us-xxxx-xx   0         0                                `\
> `ts-machineset                           1         1                                `\
> `mig-machineset                          1         1                                `

## Install Node Feature Discovery Operator

> TODO: Add preamble
> Not strictly required for GPU Operator to work, but necessary to obtain support from Nvidia

## Steps

- [ ] List the available operators for installation searching for Node Feature Discovery (NFD)

      oc get packagemanifests -n openshift-marketplace | grep nfd

> Expected output
>
> `openshift-nfd-operator                             Community Operators   8h`\
> `nfd                                                Red Hat Operators     8h`

- [ ] Apply the Namespace object

      oc apply -f infra/gpu/nfd-operator-ns.yaml

> Expected output
>
> `namespace/openshift-nfd created`

- [ ] Apply the OperatorGroup object

      oc apply -f infra/gpu/nfd-operator-group.yaml

> Expected output
>
> `operatorgroup.operators.coreos.com/nfd created`

- [ ] Apply the Subscription object

      oc apply -f infra/gpu/nfd-operator-sub.yaml

> Expected output
>
> `subscription.operators.coreos.com/nfd created`

- [ ] Verify the operator is installed and running

      oc rollout status deploy/nfd-controller-manager -n openshift-nfd --timeout=300s      

> Expected output
>
> deployment "nfd-controller-manager" successfully rolled out

> [!NOTE]
> After installing the NFD Operator, you create instance that installs the `nfd-master` and one `nfd-worker` pod for each compute node. [More Info](https://docs.openshift.com/container-platform/4.15/hardware_enablement/psap-node-feature-discovery-operator.html#Configure-node-feature-discovery-operator-sources_psap-node-feature-discovery-operator)

- [ ] Create the nfd instance object

      oc apply -f infra/gpu/nfd-instance.yaml

> Expected output
>
> `nodefeaturediscovery.nfd.openshift.io/nfd-instance created`

> [!IMPORTANT]
> The NFD Operator uses vendor PCI IDs to identify hardware in a node.

Below are some of the [PCI vendor ID assignments](https://pcisig.com/membership/member-companies?combine=10de):

| PCI id | Vendor |
| ------ | ------ |
| `10de` | NVIDIA |
| `1d0f` | AWS    |
| `1002` | AMD    |
| `8086` | Intel  |

- [ ] Verify the NFD pods are `Running` on the cluster nodes polling for devices

      oc get pods -n openshift-nfd

> Expected output
>
> `NAME                                      READY   STATUS    RESTARTS   AGE`\
> `nfd-controller-manager-xxxxxxxxxx-xxxxx   2/2     Running   0             `\
> `nfd-master-xxxxxxxxxx-xxxxx               1/1     Running   0             `\
> `nfd-worker-xxxxx                          1/1     Running   0             `\
> `nfd-worker-xxxxx                          1/1     Running   0             `\
> `nfd-worker-xxxxx                          1/1     Running   0             `

- [ ] Verify the GPU device (NVIDIA uses the PCI ID `10de`) is discovered on the GPU node. This means the NFD Operator correctly identified the node from the GPU-enabled MachineSet.

      oc describe node | egrep 'Roles|pci' | grep -v master

> Expected output
>
> `Roles:              worker`\
> `                    feature.node.kubernetes.io/pci-10de.present=true`\
> `                    feature.node.kubernetes.io/pci-1d0f.present=true`\
> `                    feature.node.kubernetes.io/pci-1d0f.present=true`\
> `Roles:              worker`\
> `                    feature.node.kubernetes.io/pci-1d0f.present=true`

## Install the NVIDIA GPU Operator

> TODO: Add preamble

## Steps

- [ ] List the available operators for installation searching for GPU

      oc get packagemanifests -n openshift-marketplace | grep gpu

> Expected output
>
> `amd-gpu-operator                                   Community Operators   8h`\
> `amd-gpu-operator                                   Certified Operators   8h`\
> `gpu-operator-certified                             Certified Operators   8h`

- [ ] Apply the Namespace object YAML file

      oc apply -f infra/gpu/nvidia-gpu-operator-ns.yaml

> Expected output
>
> `namespace/nvidia-gpu-operator created`

- [ ] Apply the OperatorGroup YAML file

      oc apply -f infra/gpu/nvidia-gpu-operator-group.yaml

> Expected output
>
> `operatorgroup.operators.coreos.com/nvidia-gpu-operator-group created`

- [ ] Apply the Subscription CR

      oc apply -f infra/gpu/nvidia-gpu-operator-subscription.yaml

> Expected output
>
> `subscription.operators.coreos.com/gpu-operator-certified created`

- [ ] Wait for Operator to finish installing

      oc rollout status deploy/gpu-operator -n nvidia-gpu-operator --timeout=300s

- [ ] Verify the Operator version

      oc get ip -n nvidia-gpu-operator

> Expected output
>
> `NAME            CSV                              APPROVAL    APPROVED`\
> `install-xxxxx   gpu-operator-certified.v24.9.2   Automatic   true`

> [!NOTE]
> The CSV version should match the latest supported [version](https://docs.nvidia.com/ai-enterprise/release-6/latest/support/support-matrix.html#supported-nvidia-infrastructure-software) of the GPU Operator.
 
- [ ] Create the cluster policy

      oc get csv -n nvidia-gpu-operator -l operators.coreos.com/gpu-operator-certified.nvidia-gpu-operator \
        -ojsonpath='{.items[0].metadata.annotations.alm-examples}' | jq '.[0]' > scratch/nvidia-gpu-clusterpolicy.json

- [ ] Apply the clusterpolicy

      oc apply -f scratch/nvidia-gpu-clusterpolicy.json

> Expected output
>
> `clusterpolicy.nvidia.com/gpu-cluster-policy created`

- [ ] Wait for the GPU Operator components to finish installing

      oc wait clusterpolicy/gpu-cluster-policy --for=condition=Ready --timeout=600s -n nvidia-gpu-operator

> [!IMPORTANT]
> This step can take up to 20 minutes to complete!

- [ ] Verify the successful installation of the NVIDIA driver

      oc get pod -l openshift.driver-toolkit -n nvidia-gpu-operator

> Expected output
>
> `NAME                                                  READY   STATUS    RESTARTS   AGE`\
> `nvidia-driver-daemonset-417.94.202503060903-0-xxxxx   2/2     Running   0             ` 

## Smoke Test

- [ ] Test GPU Access

      oc exec -n nvidia-gpu-operator $(oc get pod -n nvidia-gpu-operator -l openshift.driver-toolkit -ojsonpath='{.items[0].metadata.name}') -- nvidia-smi

> [!NOTE]
> Nvidia System Management Interface `nvidia-smi` shows memory usage, GPU utilization, and the temperature of the GPU.

- [ ] Run CUDA VectorAdd

      oc create -f infra/gpu/nvidia-gpu-sample-app.yaml -n sandbox

- [ ] Check logs

      oc logs cuda-vectoradd
