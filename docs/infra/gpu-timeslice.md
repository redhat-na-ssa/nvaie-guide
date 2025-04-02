# Time Slice

Time slicing is a mechanism to partition a GPU for more throughput.

> TODO: Add note on latency (it doesn't get *faster*)

In later sections of the guide, you will be autoscaling NIM services to deploy onto multiple GPUs. While you could scale the MachineSet to create more GPU machines, it will be more expensive.

To save on cost, let's time slice the single L4 GPU into partitions so you can test autoscaling.

> TODO: Finish this section

Next: [MIG](gpu-mig.md)
