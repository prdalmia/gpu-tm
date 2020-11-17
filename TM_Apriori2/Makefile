TM?=0

SYNC_METHOD=FGL_SYNC
CU_OBJ = apriori_kernel-fgl.o
TARGET_SUFFIX=Atomic
ifeq ($(TM),1)
SYNC_METHOD=TM_SYNC
CU_OBJ = apriori_kernel-tm.o
TARGET_SUFFIX=TM
endif

NVCC  = nvcc 
NVCC_FLAGS += -g -arch sm_20 -Xptxas -dlcm=cg -I $(NVIDIA_COMPUTE_SDK_LOCATION)/C/common/inc

GCC = g++ -g
CFLAGS  = -O2 -I /usr/include/c++ 
OBJS  = rdtsc.o 
OPTS	= -DBALT 
CUDA_INSTALL_PATH?=/usr/local/cuda
LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lm -lcudart 
LD = ld


include ../path_setup.mk

TARGET = $(BINDIR)/$(BINSUBDIR)/apriori$(TARGET_SUFFIX)

default:	$(TARGET)

clean:
	rm -f *.o $(TARGET)

$(TARGET): no_output_parapr.cpp $(OBJS) $(CU_OBJ) 
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) no_output_parapr.cpp $(OBJS) $(CU_OBJ) $(LIBS)

$(CU_OBJ): apriori_kernel.cu Itemset.cu  ListItemset.cu HashTree.cu
	$(NVCC) $(NVCC_FLAGS) -D$(SYNC_METHOD) -c  apriori_kernel.cu Itemset.cu ListItemset.cu HashTree.cu
	mv apriori_kernel.o $(CU_OBJ)

rdtsc.o: rdtsc.cpp
	$(GCC) $(CFLAGS) -c rdtsc.cpp

TimeCalculation.o: TimeCalculation.cpp
	$(GCC) $(CFLAGS) -c TimeCalculation.cpp

dec_shlib.o: dec_shlib.C dec_shlib.h
	$(GCC) $(CFLAGS) $(OPTS) -c dec_shlib.C 

dec_llsc.o: dec_llsc.s llsc.h
	cc  -c dec_llsc.s 

