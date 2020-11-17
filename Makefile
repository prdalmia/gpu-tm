
CUDA_VERSION=$(shell nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/')
noinline?=0

check_environment:
	 @if [ ! -n "$(CUDA_INSTALL_PATH)" ]; then \
		echo "ERROR *** CUDA_INSTALL_PATH not set; please set it to the location of the CUDA Toolkit."; \
		exit 201; \
	 elif [ ! -d "$(CUDA_INSTALL_PATH)" ]; then \
	 	echo "ERROR *** CUDA_INSTALL_PATH=$(CUDA_INSTALL_PATH) invalid (directory does not exist)."; \
		exit 202; \
	 elif [ ! -n "$(NVIDIA_COMPUTE_SDK_LOCATION)" ]; then \
		echo "ERROR *** NVIDIA_COMPUTE_SDK_LOCATION not set; please set it to the location of the CUDA Compute SDK"; \
		exit 203; \
	 elif [ ! -d "$(NVIDIA_COMPUTE_SDK_LOCATION)" ]; then \
	 	echo "ERROR *** NVIDIA_COMPUTE_SDK_LOCATION=$(NVIDIA_COMPUTE_SDK_LOCATION) invalid (directory does not exist)."; \
		exit 204; \
	 elif [ ! -f "$(NVIDIA_COMPUTE_SDK_LOCATION)/C/lib/libcutil_x86_64.a" -a  ! -f "$(NVIDIA_COMPUTE_SDK_LOCATION)/C/lib/libcutil.a" ]; then \
	 	echo "ERROR *** could not find $(NVIDIA_COMPUTE_SDK_LOCATION)/C/lib/libcutil_x86_64.a (or libcutil.a)"; \
	 	echo "          Build the NVIDIA GPU Computing SDK; please run Make at $(NVIDIA_COMPUTE_SDK_LOCATION)/C/common/ and at $(NVIDIA_COMPUTE_SDK_LOCATION)/shared/"; \
		exit 205; \
	 else \
		NVCC_PATH=`which nvcc`; \
		if [ $$? = 1 ]; then \
			echo ""; \
			echo "ERROR ** nvcc (from CUDA Toolkit) was not found in PATH but required to build the GPU-TM benchmarks."; \
			echo "         Try adding $(CUDA_INSTALL_PATH)/bin/ to your PATH environment variable."; \
			echo ""; \
			exit 206; \
		fi \
	fi

common: check_environment
	rm -f common; ln -s $(NVIDIA_COMPUTE_SDK_LOCATION)/C/common common;

###################################################################################################3
# GPU-TM benchmarks
###################################################################################################3

# GPUTM_BMK=$(shell ls ./)
GPUTM_BMK= barneshut  cudaCuts/CudaCutsAtomic  cudaCuts/CudaCutsTM  EA_RopaDemo  hashtable  interac  interac-nested  TM_Apriori2
$(GPUTM_BMK): common 
	make noinline=$(noinline) -C ./$@

CLEAN_GPUTM_BMK=$(addsuffix _clean,$(GPUTM_BMK))
$(CLEAN_GPUTM_BMK): common
	make clean -C ./$(patsubst %_clean,%,$@)

gputm: $(GPUTM_BMK)
clean_gputm: $(CLEAN_GPUTM_BMK)

.PHONY: common

all: gputm
cleanall: clean_gputm

