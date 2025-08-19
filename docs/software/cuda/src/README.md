# CUDA mini-workshop

```bash
for i in 1 2 4 8 16 32; do echo "# CUDA_KERNEL Threads = " $i "x" $i;nsys_easy ./03-rgb2gray -t $i; done | grep CUDA_KERNEL
```

