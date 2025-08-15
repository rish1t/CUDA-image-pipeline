CXX := nvcc
TARGET := image_processor
SRCDIR := .
OBJDIR := build

SOURCES := main.cpp ImageProcessor.cu kernels.cu
OBJECTS := $(SOURCES:%.cpp=$(OBJDIR)/%.o)
OBJECTS := $(OBJECTS:%.cu=$(OBJDIR)/%.o)

# Compiler flags
CXXFLAGS := -std=c++11 -O3
NVCCFLAGS := -arch=sm_50 -std=c++11 -O3

# OpenCV flags
OPENCV_FLAGS := $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# CUDA flags
CUDA_FLAGS := -lcudart

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(NVCCFLAGS) $^ -o $@ $(OPENCV_FLAGS) $(CUDA_FLAGS)

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	$(CXX) $(NVCCFLAGS) -c $< -o $@ $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET) *.png

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Example usage
test: $(TARGET)
	@echo "To test the program, run:"
	@echo "./$(TARGET) <input_image.jpg>"