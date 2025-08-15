#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void rgb_to_grayscale_kernel(const uchar3* input, unsigned char* output, 
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = input[idx];
        // Standard luminance conversion coefficients
        output[idx] = (unsigned char)(0.299f * pixel.z + 0.587f * pixel.y + 0.114f * pixel.x);
    }
}

__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, 
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        const float kernel[3][3] = {
            {1.0f/16, 2.0f/16, 1.0f/16},
            {2.0f/16, 4.0f/16, 2.0f/16},
            {1.0f/16, 2.0f/16, 1.0f/16}
        };

        float sum = 0.0f;
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * kernel[ky + 1][kx + 1];
            }
        }
        output[y * width + x] = (unsigned char)sum;
    } else if (x < width && y < height) {
        output[y * width + x] = input[y * width + x];
    }
}

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output, 
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X gradient
        int gx = -input[(y-1)*width + (x-1)] - 2*input[y*width + (x-1)] - input[(y+1)*width + (x-1)] +
                  input[(y-1)*width + (x+1)] + 2*input[y*width + (x+1)] + input[(y+1)*width + (x+1)];

        // Sobel Y gradient  
        int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)] +
                  input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];

        float magnitude = sqrtf((float)(gx*gx + gy*gy));
        output[y * width + x] = (unsigned char)fminf(magnitude * 1.5f, 255.0f);
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}