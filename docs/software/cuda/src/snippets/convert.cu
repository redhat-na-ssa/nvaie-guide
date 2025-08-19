    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int index = (y * width + x);
        index = (x * height + y);
        unsigned char r = d_r[index];
        unsigned char g = d_g[index];
        unsigned char b = d_b[index];

        // Standard NTSC grayscale conversion formula
        d_gray[index] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

    }
