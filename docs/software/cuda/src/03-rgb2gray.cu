//
// CUDA program to convert RGB image to grayscale.
//
//
#include "rgb2gray.hpp"
#include <png.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cstdlib>
using namespace std;

//
// TODO #1
// Add the CUDA Kernel Device code to convert the RGB image to grayscale.
//
//
__global__ void convert(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height)
{

}

__host__ std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> allocateDeviceMemory(int width, int height)
{
    cout << "Allocating GPU device memory\n";
    int num_image_pixels = width * height;
    size_t size = num_image_pixels * sizeof(unsigned char);

    // Allocate the device input vector d_r
    unsigned char *d_r = NULL;
    cudaError_t err = cudaMalloc(&d_r, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_g
    unsigned char *d_g = NULL;
    err = cudaMalloc(&d_g, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_b
    unsigned char *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_gray
    unsigned char *d_gray = NULL;
    err = cudaMalloc(&d_gray, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_gray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate device constant symbols for width and height
    cudaMemcpyToSymbol(d_width, &width, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_height, &height, sizeof(int), 0, cudaMemcpyHostToDevice);

    return {d_r, d_g, d_b, d_gray};
}


__host__ void copyFromHostToDevice(unsigned char *h_r, unsigned char *d_r, unsigned char *h_g, unsigned char *d_g, unsigned char *h_b, unsigned char *d_b, int width, int height)
{
    cout << "Copying from Host to Device\n";
    int num_image_pixels = width * height;
    size_t size = num_image_pixels * sizeof(unsigned char);

    cudaError_t err;
    err = cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector b from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void executeKernel(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height, int threadsPerBlock)
{
    cout << "Executing kernel\n";

    // Calculate grid and block dimensions
    dim3 block(threadsPerBlock, threadsPerBlock, 1);
    dim3 grid(ceil((float) width / block.x), ceil((float) height / block.y), 1);

    cout << "threadsPerBlock = " << threadsPerBlock << " x " << threadsPerBlock << endl;
    cout << "grid = " << grid.x << " " << grid.y << ", " << "block = " << block.x << " " << block.y << endl;
    cout << "total pixels = " << width * height << ", total threads = " << grid.x * grid.y * block.x * block.y << endl;
    
    // 
    // TODO #2 
    // Call the CUDA kernel that converts the RGB image to gray scale.
    //

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void copyFromDeviceToHost(unsigned char *d_gray, unsigned char *gray, int width, int height)
{
    cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = width * height * sizeof(unsigned char);

    cudaError_t err = cudaMemcpy(gray, d_gray, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_gray from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray)
{
    cout << "Deallocating GPU device memory\n";
    cudaError_t err = cudaFree(d_r);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_gray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device int variable d_image_num_pixels (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 16;
    std::string inputImage = "images/rainbow_2048.png";
    std::string outputImage = "grey.png";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
    }

    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " threadsPerBlock dimension: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, threadsPerBlock};
}

// Helper: Read PNG file into RGB arrays
std::tuple<int, int, std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>>
readPNG(const std::string &filename)
{
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) { cerr << "Cannot open file " << filename << endl; exit(1); }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) { cerr << "PNG read error\n"; exit(1); }
    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    int alpha = png_get_valid(png, info, PNG_INFO_tRNS) ? 1 : 0;
    int depth = png_get_bit_depth(png, info);
    cout << "Image width: " << width << ", height: " << height << ", depth: " << depth << ", alpha: " << alpha << endl;

    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    int rowbytes = png_get_rowbytes(png, info);
    int channels = rowbytes / width; // 3 for RGB, 4 for RGBA

    std::vector<unsigned char> image_data(rowbytes * height);
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++)
        row_pointers[y] = &image_data[y * rowbytes];

    png_read_image(png, row_pointers.data());
    fclose(fp);
    png_destroy_read_struct(&png, &info, nullptr);

    std::vector<unsigned char> r(width * height), g(width * height), b(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int base = x * channels;
            r[idx] = row_pointers[y][base + 0];
            g[idx] = row_pointers[y][base + 1];
            b[idx] = row_pointers[y][base + 2];
            // Ignore alpha channel if present
        }
    }
    return {width, height, r, g, b};
}

// Helper: Write grayscale PNG file
void writePNG(const std::string &filename, const unsigned char *gray, int width, int height)
{
    FILE *fp = fopen(filename.c_str(), "wb");
    if (!fp) { cerr << "Cannot open file " << filename << endl; exit(1); }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) { cerr << "PNG write error\n"; exit(1); }
    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png, info);

    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++)
        row_pointers[y] = (png_bytep)(gray + y * width); // FIXED: correct row pointer

    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);

    fclose(fp);
    png_destroy_write_struct(&png, &info);
}

// ...existing CUDA memory and utility functions...

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<2>(parsedCommandLineArgsTuple);

    try 
    {
        auto[width, height, h_r_vec, h_g_vec, h_b_vec] = readPNG(inputImage);
        unsigned char *h_r = h_r_vec.data();
        unsigned char *h_g = h_g_vec.data();
        unsigned char *h_b = h_b_vec.data();

        unsigned char *gray = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
        std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> memoryTuple = allocateDeviceMemory(width, height);
        unsigned char *d_r = get<0>(memoryTuple);
        unsigned char *d_g = get<1>(memoryTuple);
        unsigned char *d_b = get<2>(memoryTuple);
        unsigned char *d_gray = get<3>(memoryTuple);

        copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, width, height);

        executeKernel(d_r, d_g, d_b, d_gray, width, height, threadsPerBlock);

        copyFromDeviceToHost(d_gray, gray, width, height);
        deallocateMemory(d_r, d_g, d_b, d_gray);
        cleanUpDevice();

        writePNG(outputImage, gray, width, height); // FIXED: correct order
    }
    catch (std::exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}
