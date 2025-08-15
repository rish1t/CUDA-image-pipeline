# CUDA Image Processing Pipeline

A high-performance image processing pipeline implemented in CUDA C++, featuring parallel RGB to grayscale conversion, Gaussian blur filtering, and Sobel edge detection.

## Features

- **Parallel RGB to Grayscale Conversion** - Optimized CUDA kernel using standard luminance coefficients
- **Gaussian Blur Filter** - 3x3 kernel with proper boundary handling
- **Sobel Edge Detection** - Gradient-based edge detection with magnitude enhancement
- **Object-Oriented Design** - Clean C++ class structure with RAII memory management
- **OpenCV Integration** - Seamless image I/O and format support

## Architecture

The project is structured into modular components:

- `kernels.cu/.cuh` - CUDA kernel implementations and declarations
- `ImageProcessor.hpp/.cu` - Main processing class with GPU memory management
- `main.cpp` - Clean command-line interface
- `Makefile` - Professional build system with optimization flags

## Requirements

- NVIDIA GPU with CUDA Compute Capability 5.0+
- CUDA Toolkit 10.0+
- OpenCV 3.0+ (with development headers)
- C++11 compatible compiler

## Building

```bash
make clean
make
```

For installation to system path:
```bash
make install
```

## Usage

```bash
./image_processor input_image.jpg
```

**Output Files:**
- `step1_grayscale.png` - Grayscale conversion result
- `step2_blur.png` - Gaussian blur applied
- `step3_edges.png` - Final edge detection output

## Performance Features

- **Memory Coalescing** - Optimized memory access patterns for maximum bandwidth
- **Thread Block Optimization** - 16x16 thread blocks for optimal occupancy
- **Boundary Handling** - Efficient edge case management in convolution operations
- **RAII Memory Management** - Automatic cleanup prevents memory leaks

## Technical Details

### Grayscale Conversion
Uses standard ITU-R BT.709 luminance coefficients: `Y = 0.299R + 0.587G + 0.114B`

### Gaussian Blur
3x3 kernel with normalized coefficients for noise reduction while preserving edges.

### Sobel Edge Detection
Combines horizontal and vertical gradient calculations with magnitude enhancement for robust edge detection.

## Project Structure

```
├── main.cpp              # Application entry point
├── ImageProcessor.hpp    # Class declaration
├── ImageProcessor.cu     # Class implementation
├── kernels.cuh          # CUDA kernel headers
├── kernels.cu           # CUDA kernel implementations
├── Makefile             # Build configuration
└── README.md            # Documentation
```