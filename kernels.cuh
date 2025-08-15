#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const uchar3* input, unsigned char* output, 
                                       int width, int height);

__global__ void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, 
                                   int width, int height);

__global__ void sobel_kernel(const unsigned char* input, unsigned char* output, 
                           int width, int height);

#endif