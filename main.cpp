#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageProcessor.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }

    try {
        cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (input_image.empty()) {
            std::cerr << "Error: Cannot load image " << argv[1] << std::endl;
            return 1;
        }

        std::cout << "Processing image: " << argv[1] 
                  << " (" << input_image.cols << "x" << input_image.rows << ")" << std::endl;

        ImageProcessor processor(input_image);
        
        processor.convert_to_grayscale();
        processor.save_grayscale("step1_grayscale.png");
        std::cout << "✓ Grayscale conversion complete" << std::endl;
        
        processor.apply_gaussian_blur();
        processor.save_blurred("step2_blur.png");
        std::cout << "✓ Gaussian blur applied" << std::endl;
        
        processor.detect_edges();
        processor.save_edges("step3_edges.png");
        std::cout << "✓ Edge detection complete" << std::endl;
        
        std::cout << "\nProcessing pipeline finished successfully!" << std::endl;
        std::cout << "Output files: step1_grayscale.png, step2_blur.png, step3_edges.png" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}