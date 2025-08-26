//
// CUDA program to convert RGB image to grayscale.
//
//
#include "rgb2gray.hpp"
#include <Magick++.h> 
#include <iostream>   
using namespace std;
using namespace Magick;

//
// TODO #1
// Add the CUDA Kernel Device code to convert the RGB image to grayscale.
//
//
__global__ void convert(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height)
{

}

__host__ std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> allocateDeviceMemory(int rows, int columns)
{
    cout << "Allocating GPU device memory\n";
    int num_image_pixels = rows * columns;
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

    // Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);

    return {d_r, d_g, d_b, d_gray};
}


__host__ void copyFromHostToDevice(unsigned char *h_r, unsigned char *d_r, unsigned char *h_g, unsigned char *d_g, unsigned char *h_b, unsigned char *d_b, int rows, int columns)
{
    cout << "Copying from Host to Device\n";
    int num_image_pixels = rows * columns;
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

__host__ void executeKernel(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int rows, int columns, int threadsPerBlock)
{
    cout << "Executing kernel\n";

    // Calculate grid and block dimensions
    dim3 block(threadsPerBlock, threadsPerBlock, 1);
    dim3 grid(ceil((float) columns / block.x), ceil((float) rows / block.y), 1);

    cout << "threadsPerBlock = " << threadsPerBlock << " x " << threadsPerBlock << endl;
    cout << "grid = " << grid.x << " " << grid.y << ", " << "block = " << block.x << " " << block.y << endl;
    cout << "total pixels = " << rows * columns << ", total threads = " << grid.x * grid.y * block.x * block.y << endl;
    
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

__host__ void copyFromDeviceToHost(unsigned char *d_gray, unsigned char *gray, int rows, int columns)
{
    cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = rows * columns * sizeof(unsigned char);

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

__host__ std::tuple<int, int, unsigned char *, unsigned char *, unsigned char *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Image img;
    img.read(inputFile);

    const int rows = img.rows();
    const int columns = img.columns();
    const int channels = img.depth();

    cout << "Rows: " << rows << " Columns: " << columns << " Channels: " << channels << "\n";

    unsigned char *h_r = (unsigned char *)malloc(sizeof(unsigned char) * rows * columns);
    unsigned char *h_g = (unsigned char *)malloc(sizeof(unsigned char) * rows * columns);
    unsigned char *h_b = (unsigned char *)malloc(sizeof(unsigned char) * rows * columns);
    cout << "malloc passed" << endl;
    
    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            Color pixel = img.pixelColor(c, r);
            h_r[r*rows+c] = (int) 255 * (float) pixel.redQuantum() / QuantumRange;
            h_g[r*rows+c] = (int) 255 * (float) pixel.greenQuantum() / QuantumRange;
            h_b[r*rows+c] = (int) 255 * (float) pixel.blueQuantum() / QuantumRange;
        }
    }

    cout << "Finished reading image into RGB arrays\n";

    return {rows, columns, h_r, h_g, h_b};
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<2>(parsedCommandLineArgsTuple);
    try 
    {
        auto[rows, columns, h_r, h_g, h_b] = readImageFromFile(inputImage);
        cout << "Finshed reading image file." << endl;

        unsigned char *gray = (unsigned char *)malloc(sizeof(unsigned char) * rows * columns);
        std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> memoryTuple = allocateDeviceMemory(rows, columns);
        unsigned char *d_r = get<0>(memoryTuple);
        unsigned char *d_g = get<1>(memoryTuple);
        unsigned char *d_b = get<2>(memoryTuple);
        unsigned char *d_gray = get<3>(memoryTuple);

        copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, rows, columns);

        executeKernel(d_r, d_g, d_b, d_gray, rows, columns, threadsPerBlock);

        copyFromDeviceToHost(d_gray, gray, rows, columns);
        deallocateMemory(d_r, d_g, d_b, d_gray);
        cleanUpDevice();

        InitializeMagick(*argv);
        Image image;
        // Each pixel is GRAY, 8 bytes (4 unsigned shorts per pixel).
        vector<unsigned short> rawPixels(columns * rows * 4); 

        for (size_t i = 0; i < columns * rows; ++i) {
            rawPixels[i * 4 + 0] = gray[i] * 257; // Red
            rawPixels[i * 4 + 1] = gray[i] * 257; // Green
            rawPixels[i * 4 + 2] = gray[i] * 257; // Blue
            rawPixels[i * 4 + 3] = 65535; // Alpha (fully opaque)
        }
        
        // Create an Image from the gray data.
        Blob my_blob(rawPixels.data(), rawPixels.size() * sizeof(unsigned short));
        image.size(Geometry(columns, rows));
	    // Specify the pixel format (Red, Green, Blue, Alpha)
        image.magick("RGBA"); 
        image.read(my_blob);
	    // Convert the image from rgba to png.
	    image.magick("PNG"); 

        cout << "Writing output image to: " << outputImage << endl;
        image.write(outputImage);
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}
