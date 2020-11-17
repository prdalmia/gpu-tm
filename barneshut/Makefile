
export TM?=1
ifeq ($(TM),1)
EXECUTABLE  := barneshut
CUFILES_sm_13 := BurtscherGCG.TM.cu
else
EXECUTABLE	:= barneshut-orig
CUFILES_sm_13 := BurtscherGCG.1.1.cu
endif

LINKFLAGS += -g
COMMONFLAGS += -g -Xptxas -dlcm=cg  

################################################################################
# Rules and targets

include ../common_wrapper.mk

