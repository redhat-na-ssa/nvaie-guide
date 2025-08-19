//
// Display the CUDA device properties
//
// See https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
//
#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
       if (nDevices < 0) {
       printf("There is no device supporting CUDA\n");
       return 0;
       }
       
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  totalGlobalMem = %3.3f (GB)\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  multiProcessorCount = %d\n", prop.multiProcessorCount);
    //
    // TODO #1
    //
    // Add code to display the CUDA device maximum threads per multi processor property.
    // Refer to https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
    //
    printf("  maxThreadsPerMultiProcessor = %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  maxBlocksPerMultiProcessor = %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsDim = %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  maxGridSize = %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  L2 Cache Size (MB): %d\n", prop.l2CacheSize / (1024 * 1024));

  }
}

