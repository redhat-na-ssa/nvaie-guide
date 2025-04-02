# Prerequisites

## Red Hat

Request an environment from the Demo Catalog system.

- [AWS with OpenShift Open Environment](https://catalog.demo.redhat.com/catalog?item=babylon-catalog-prod/sandboxes-gpte.sandbox-ocp.prod&utm_source=webapp&utm_medium=share-link)
    - Activity: `Practice / Enablement`
    - Purpose: `Trying out a technical solution`
    - Region: `us-east-2`
    - OpenShift Version: `4.17`
    - Control Plane Count: `1`
    - Control Plane Instance Type: `m6a.4xlarge`

Login to the cluster with your credentials.

This is a single node OCP cluster. Let's scale the cluster for a little more room.

```bash
MACHINESET=$(oc get machineset -n openshift-machine-api -o jsonpath='{.items[0].metadata.name}')
oc scale machineset $MACHINESET -n openshift-machine-api --replicas=1
```

## Nvidia

Create a Nvidia account using this [link](https://www.nvidia.com/en-us/account/). 

After creating your account, create a Personal API key using this [link](https://org.ngc.nvidia.com/setup/api-keys).

Include `NGC Catalog` under *Services Included*.

Set this key in your terminal:

```bash
export NGC_API_KEY=
```

## Terminal

Clone this repository:

```bash
git clone --depth 1 \
  https://github.com/redhat-na-ssa/nvaie-guide.git
``` 

Change into the `nvaie-guide` directory:

```bash
cd nvaie-guide
```

Create a scratch directory:

```bash
mkdir scratch
```

You are going to use the official Nvidia Helm charts for some components such as Nvidia RIVA.

> TODO: Add link to Riva section

Follow the Helm [docs](https://helm.sh/docs/intro/install/) to install Helm.

Next: [GPU Operator](infra/gpu-operator.md)
