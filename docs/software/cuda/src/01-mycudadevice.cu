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

  }
}

