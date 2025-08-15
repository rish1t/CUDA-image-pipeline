#include "ImageProcessor.hpp"
#include "kernels.cuh"
#include <iostream>
#include <stdexcept>

ImageProcessor::ImageProcessor(const cv::Mat& input_image) 
    : width_(input_image.cols), height_(input_image.rows) {
    
    gray_size_ = width_ * height_ * sizeof(unsigned char);
    color_size_ = width_ * height_ * sizeof(uchar3);
    
    allocate_memory();
    
    // Copy input image to device
    cudaError_t err = cudaMemcpy(d_color_, input_image.ptr<uchar3>(), 
                                color_size_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy image to GPU memory");
    }
}

ImageProcessor::~ImageProcessor() {
    cleanup_memory();
}

void ImageProcessor::allocate_memory() {
    // Allocate host memory
    h_gray_ = (unsigned char*)malloc(gray_size_);
    h_blurred_ = (unsigned char*)malloc(gray_size_);
    h_output_ = (unsigned char*)malloc(gray_size_);
    
    // Allocate device memory
    cudaMalloc(&d_color_, color_size_);
    cudaMalloc(&d_gray_, gray_size_);
    cudaMalloc(&d_blurred_, gray_size_);
    cudaMalloc(&d_output_, gray_size_);
}

void ImageProcessor::cleanup_memory() {
    // Free device memory
    cudaFree(d_color_);
    cudaFree(d_gray_);
    cudaFree(d_blurred_);
    cudaFree(d_output_);
    
    // Free host memory
    free(h_gray_);
    free(h_blurred_);
    free(h_output_);
}

dim3 ImageProcessor::get_grid_size(dim3 block_size) {
    return dim3((width_ + block_size.x - 1) / block_size.x, 
                (height_ + block_size.y - 1) / block_size.y);
}

void ImageProcessor::convert_to_grayscale() {
    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);
    
    rgb_to_grayscale_kernel<<<grid_size, block_size>>>(d_color_, d_gray_, width_, height_);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_gray_, d_gray_, gray_size_, cudaMemcpyDeviceToHost);
}

void ImageProcessor::apply_gaussian_blur() {
    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);
    
    gaussian_blur_kernel<<<grid_size, block_size>>>(d_gray_, d_blurred_, width_, height_);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_blurred_, d_blurred_, gray_size_, cudaMemcpyDeviceToHost);
}

void ImageProcessor::detect_edges() {
    dim3 block_size(16, 16);
    dim3 grid_size = get_grid_size(block_size);
    
    sobel_kernel<<<grid_size, block_size>>>(d_blurred_, d_output_, width_, height_);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output_, d_output_, gray_size_, cudaMemcpyDeviceToHost);
}

void ImageProcessor::save_grayscale(const std::string& filename) {
    cv::imwrite(filename, cv::Mat(height_, width_, CV_8UC1, h_gray_));
}

void ImageProcessor::save_blurred(const std::string& filename) {
    cv::imwrite(filename, cv::Mat(height_, width_, CV_8UC1, h_blurred_));
}

void ImageProcessor::save_edges(const std::string& filename) {
    cv::imwrite(filename, cv::Mat(height_, width_, CV_8UC1, h_output_));
}

cv::Mat ImageProcessor::get_grayscale_image() {
    return cv::Mat(height_, width_, CV_8UC1, h_gray_).clone();
}

cv::Mat ImageProcessor::get_blurred_image() {
    return cv::Mat(height_, width_, CV_8UC1, h_blurred_).clone();
}

cv::Mat ImageProcessor::get_edge_image() {
    return cv::Mat(height_, width_, CV_8UC1, h_output_).clone();
}