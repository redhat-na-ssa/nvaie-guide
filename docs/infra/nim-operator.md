# NIM (Nvidia Inference Microservices) Operator

## Create NIM Operator

List the available operators for installation searching for `nim-operator`:

```bash
oc get packagemanifests -n openshift-marketplace | grep nim-operator
```

```text
nim-operator-certified                           Certified Operators   32h
```

Create the NIM operator:

```bash
oc create -f configs/infra/nim/nim-operator.yaml
```

Use the `rollout` command to verify the deployment. You might get an error if the deployment does not exist yet. In that case, wait a few seconds and try again.

Wait for the operator deployment:

```bash
oc rollout status deploy/k8s-nim-operator -n openshift-operators --timeout=300s
```

```text
deployment "k8s-nim-operator" successfully rolled out
```

Verify the version of the NIM operator that was installed:

```bash
oc get ip -n openshift-operators
```

```text
NAME            CSV                             APPROVAL    APPROVED
install-xxxxx   nim-operator-certified.v2.0.0   Automatic   true
```

> [!NOTE]
> The CSV version should match the latest supported [version](https://docs.nvidia.com/ai-enterprise/release-6/latest/support/support-matrix.html#supported-nvidia-configs/infrastructure-software) of the NIM Operator.

