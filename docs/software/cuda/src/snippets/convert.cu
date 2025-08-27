    //
    // Calculate the global pixel coordinates for this thread.
    //
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //
    // Check if the pixel coordinates are within the image bounds.
    //
    if (x < width && y < height) {
        int index = y * width + x;
        unsigned char r = d_r[index];
        unsigned char g = d_g[index];
        unsigned char b = d_b[index];
        // Convert to grayscale using the Standard NTSC conversion formula.
        d_gray[index] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
