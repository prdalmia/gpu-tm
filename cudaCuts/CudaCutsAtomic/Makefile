EXECUTABLE	:= cudaCutsAtomic
CUFILES_sm_13 := Example.cu
CCFILES		:= \
#	imageWrite.cpp
	
CU_DEPS := CudaCuts.cu CudaCuts.h CudaCuts_kernel.cu Example.h #edgeWeight.cu PushRelabel_kernel.cu Bfs_kernel.cu
CUDACCFLAGS += -g -Xptxas -dlcm=cg # -v -g

include ../../common_wrapper.mk

