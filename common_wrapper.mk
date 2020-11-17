# Wrapper for common.mk from NVIDIA COMPUTE SDK 
# Usage: include this file in the benchmark's Makefile 

BMKSRCDIR:=$(dir $(lastword $(MAKEFILE_LIST)))

include $(BMKSRCDIR)/path_setup.mk
include $(BMKSRCDIR)/common/common.mk

.PHONY:testpaths
testpaths: 
	@echo "CUDA_VERSION=$(CUDA_VERSION)"
	@echo "BMKSRCDIR=$(BMKSRCDIR)"
	@echo "ROOTDIR=$(ROOTDIR)"
	@echo "BINDIR=$(BINDIR)"

