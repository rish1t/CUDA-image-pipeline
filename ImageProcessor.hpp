#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

class ImageProcessor {
private:
    int width_, height_;
    size_t gray_size_, color_size_;
    
    // Device memory pointers
    uchar3* d_color_;
    unsigned char* d_gray_;
    unsigned char* d_blurred_;
    unsigned char* d_output_;
    
    // Host memory pointers
    unsigned char* h_gray_;
    unsigned char* h_blurred_;
    unsigned char* h_output_;
    
    void allocate_memory();
    void cleanup_memory();
    dim3 get_grid_size(dim3 block_size);

public:
    ImageProcessor(const cv::Mat& input_image);
    ~ImageProcessor();
    
    void convert_to_grayscale();
    void apply_gaussian_blur();
    void detect_edges();
    
    void save_grayscale(const std::string& filename);
    void save_blurred(const std::string& filename);
    void save_edges(const std::string& filename);
    
    cv::Mat get_grayscale_image();
    cv::Mat get_blurred_image();
    cv::Mat get_edge_image();
};

#endif