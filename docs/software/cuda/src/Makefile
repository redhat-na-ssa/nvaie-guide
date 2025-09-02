# 
# Makefile for CUDA and ImageMagick on RHEL
#

# Set paths and flags	
CUDA_PATH   := /usr/local/cuda
NVCC        := $(CUDA_PATH)/bin/nvcc
CXX         := g++
CXXFLAGS    := --std=c++17 -Wno-deprecated-gpu-targets
LDFLAGS     := -lpng

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