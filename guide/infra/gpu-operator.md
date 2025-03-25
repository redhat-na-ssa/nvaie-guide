# GPU Operator

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
> `cluster-xxxxx-xxxxx-worker-us-xxxx-xx   0         0                                `

## Install Node Feature Discovery Operator

## Install Nvidia GPU Operator


