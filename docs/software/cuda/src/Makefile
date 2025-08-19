# 
# Makefile for CUDA and ImageMagick on RHEL
#
# yum install libomp-devel ImageMagick-c++-devel
#
# g++ `Magick++-config --cxxflags --cppflags` -o readWriteImages readWriteImages.cpp `Magick++-config --ldflags --libs`
# nvcc does not support the -fopenmp flag
#

# Set paths and flags	
CUDA_PATH   := /usr/local/cuda
NVCC        := $(CUDA_PATH)/bin/nvcc
CXXFLAGS += --std c++17 -I/usr/include/ImageMagick-6 -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -Wno-deprecated-gpu-targets 
LDFLAGS += `Magick++-config --ldflags --libs`

# CUDA sources
CUDA_SRCS = $(wildcard *.cu)
CUDA_BINS = $(CUDA_SRCS:.cu=)

all: $(CUDA_BINS)

%: %.cu
	$(NVCC) $(CXXFLAGS)$(LDFLAGS) $< -o $@

clean:
	rm -f $(CUDA_BINS) gray.png nsys_easy.*

.PHONY: all clean