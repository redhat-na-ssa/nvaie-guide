# 
# Makefile for CUDA and ImageMagick on RHEL
#

# Set paths and flags	
CUDA_PATH   := /usr/local/cuda
NVCC        := $(CUDA_PATH)/bin/nvcc
CXX         := g++
CXXFLAGS    := --std=c++17 -I/usr/include/ImageMagick-6 -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -Wno-deprecated-gpu-targets
LDFLAGS     := $(shell Magick++-config --ldflags --libs) -lpng

# CUDA sources
CUDA_SRCS   := $(wildcard *.cu)
CUDA_BINS   := $(CUDA_SRCS:.cu=)

all: $(CUDA_BINS)

# Build CUDA binaries
%: %.cu 
	$(NVCC) $(CXXFLAGS) $< $(LDFLAGS) -o $@

clean:
	rm -f $(CUDA_BINS) grey.png nsys_easy.*

.PHONY: all clean