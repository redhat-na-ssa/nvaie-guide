# NIM (Nvidia Inference Microservices) Operator

> TODO: Add preamble\
> TODO: Add [supported models](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html)

## Create NIM Operator

> [!IMPORTANT]
> TODO

## Steps

- [ ] List the available operators for installation searching for `nim-operator`

      oc get packagemanifests -n openshift-marketplace | grep nim-operator

> Expected output
>
> `nim-operator-certified                           Certified Operators   23h`

- [ ] Create the NIM Operator Subscription

      oc create -f configs/infra/nim/nim-operator-sub.yaml

> Expected output
>
> `subscription.operators.coreos.com/nim-operator-certified created`

- [ ] Wait for Operator to finish installing

      oc rollout status deploy/k8s-nim-operator -n openshift-operators --timeout=300s

> Expected output
> 
> `deployment "k8s-nim-operator" successfully rolled out`

- [ ] Verify the Operator version

      oc get ip -n openshift-operators

> Expected output
>
> `NAME            CSV                              APPROVAL    APPROVED`\
> `install-xxxxx   nim-operator-certified.v1.0.1    Automatic   true`

> [!NOTE]
> The CSV version should match the latest supported [version](https://docs.nvidia.com/ai-enterprise/release-6/latest/support/support-matrix.html#supported-nvidia-configs/infrastructure-software) of the NIM Operator.

