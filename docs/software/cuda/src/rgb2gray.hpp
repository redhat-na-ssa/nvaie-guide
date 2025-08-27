#include <iostream>
#include <stdio.h>
#include <tuple>
#include <string>

using namespace std;

__device__ __constant__ int d_width;
__device__ __constant__ int d_height;

__host__ std::tuple<std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[]);
__host__ std::tuple<int, int, unsigned char *, unsigned char *, unsigned char *> readImageFromFile(std::string inputFile);
__host__ void copyFromDeviceToHost(unsigned char *d_gray, unsigned char *gray, int width, int height);
__host__ std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> allocateDeviceMemory(int width, int height);
__host__ void deallocateMemory(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, int *d_image_num_pixels);
__host__ void cleanUpDevice();
__host__ void copyFromHostToDevice(unsigned char *h_r, unsigned char *d_r, unsigned char *h_g, unsigned char *d_g, unsigned char *h_b, unsigned char *d_b, int width, int height);
__host__ void executeKernel(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height, int threadsPerBlock);
__global__ void convert(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height);
