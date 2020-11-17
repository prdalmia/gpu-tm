# Perform setup to use common.mk from NVIDIA COMPUTE SDK 
# Usage: include this file in the benchmark's Makefile before common.mk 

CUDA_VERSION=$(shell nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/')
BINDIR=$(shell pwd | sed -re 's/simulations.*/simulations/')/bin/$(CUDA_VERSION)
BINSUBDIR=release
ROOTDIR=$(NVIDIA_COMPUTE_SDK_LOCATION)/C/src/
ROOTOBJDIR=obj_$(CUDA_VERSION)
SETENV=export BOOST_LIB=/usr/lib64; \
       export BOOST_ROOT=/usr/include; \
       export BOOST_VER=""; \
       export OPENMPI_BINDIR=/usr/lib64/mpi/gcc/openmpi/bin/; 

