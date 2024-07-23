
# Build directories
DFFT_BUILD_DIR ?= build
DFFT_LIB_DIR ?= lib

# Output library
DFFT_AR ?= libswfft.a

# Source directories
DFFT_TEST_DIR ?= test
DFFT_SOURCE_DIR ?= src
DFFT_INCLUDE_DIR ?= include

# Platform
DFFT_PLATFORM ?= unknown

# CUDA settings
CUDA_PATH ?= /usr/local/cuda
DFFT_CUDA_LIB ?= $(CUDA_PATH)/lib64
DFFT_CUDA_INC ?= $(CUDA_PATH)/include
DFFT_CUDA_ARCH ?= -gencode=arch=compute_60,code=sm_60
DFFT_CUDA_LD ?= -lcufft -lcudart
DFFT_CUDA_FLAGS ?= -lineinfo -Xptxas -v -Xcompiler="-fPIC"

# CUDA MPI settings
USE_CUDAMPI ?= FALSE
ifeq ($(USE_CUDAMPI), TRUE)
DFFT_CUDA_MPI ?=
else
DFFT_CUDA_MPI ?= -DSWFFT_NOCUDAMPI
endif

# Include/link settings
DFFT_INCLUDE ?= -I$(DFFT_INCLUDE_DIR) -I$(DFFT_CUDA_INC) # -I$(DFFT_SOURCE_DIR)
DFFT_MPI_INCLUDE ?= -I/usr/include/x86_64-linux-gnu/mpich
DFFT_LD ?= -L$(DFFT_CUDA_LIB)

# GPU settings
DFFT_GPU ?= CUDA
USE_GPU ?= TRUE
ifeq ($(USE_GPU), TRUE)
GPU_FLAG = -DSWFFT_GPU
else
GPU_FLAG =
endif

# Backend settings
FFT_BACKEND ?= CUFFT FFTW
DIST_BACKEND ?= ALLTOALL PAIRWISE HQFFT
DFFT_FFT_BACKEND_DEFINES ?= $(FFT_BACKEND:%=-DSWFFT_%)
DFFT_DIST_BACKEND_DEFINES ?= $(DIST_BACKEND:%=-DSWFFT_%)

# FFTW settings
DFFT_FFTW_HOME ?= $(shell dirname $(shell dirname $(shell which fftw-wisdom)))
DFFT_FFTW_CPPFLAGS ?= -I$(DFFT_FFTW_HOME)/include

# OMP settings
USE_OMP ?= FALSE
ifeq ($(USE_OMP), TRUE)
DFFT_FFTW_LDFLAGS ?= -L$(DFFT_FFTW_HOME)/lib -lfftw3_omp -lfftw3 -lfftw3f -lm
DFFT_OPENMP ?= -fopenmp
else
DFFT_FFTW_LDFLAGS ?= -L$(DFFT_FFTW_HOME)/lib -lfftw3 -lfftw3f -lm
DFFT_OPENMP ?=
endif

# Compilers
DFFT_MPI_CC ?= mpicc -O3
DFFT_MPI_CXX ?= mpicxx -O3 -Wall -Wpedantic -Werror
DFFT_CUDA_CC ?= nvcc -O3

# Source files/objects
ifeq ($(USE_GPU), TRUE)
SOURCES := $(shell find $(DFFT_SOURCE_DIR) -name '*.cpp') $(shell find $(DFFT_SOURCE_DIR) -name '*.cu')
OBJECTS := $(SOURCES:%.cpp=%.o)
OBJECTS := $(OBJECTS:%.cu=%.o)
OUTPUTS := $(OBJECTS:%=$(DFFT_BUILD_DIR)/%)
else
SOURCES := $(shell find $(DFFT_SOURCE_DIR) -name '*.cpp')
OBJECTS := $(SOURCES:%.cpp=%.o)
OUTPUTS := $(OBJECTS:%=$(DFFT_BUILD_DIR)/%)
endif

# Test files/objects
TESTSOURCES := $(shell find $(DFFT_TEST_DIR) -name '*.cpp') $(shell find $(DFFT_TEST_DIR) -name '*.cu')
TESTOBJECTS_1 := $(TESTSOURCES:%.cpp=%.o)
TESTOBJECTS := $(TESTOBJECTS_1:%.cu=%.o)

.PHONY: main
main: $(DFFT_BUILD_DIR)/testdfft $(DFFT_BUILD_DIR)/benchmark $(DFFT_BUILD_DIR)/testks $(DFFT_BUILD_DIR)/testalltoallgpu $(DFFT_BUILD_DIR)/harness

.secondary: $(OUTPUTS) $(TESTOBJECTS)

$(DFFT_BUILD_DIR)/%: $(DFFT_BUILD_DIR)/$(DFFT_TEST_DIR)/%.o $(DFFT_LIB_DIR)/$(DFFT_AR)
	mkdir -p $(@D)
	$(DFFT_MPI_CXX) $(GPU_FLAG) $(DFFT_CUDA_MPI) -DSWFFT_$(DFFT_GPU) -DSWFFT_PLATFORM=$(DFFT_PLATFORM) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $^ $(DFFT_FFTW_LDFLAGS) $(DFFT_OPENMP) -L$(DFFT_CUDA_LIB) $(DFFT_CUDA_LD) -fPIC -o $@

.PHONY: clean
clean:
	rm -rf $(DFFT_BUILD_DIR)
	rm -rf $(DFFT_LIB_DIR)

$(DFFT_BUILD_DIR)/%.o: %.cpp
	mkdir -p $(@D)
	$(DFFT_MPI_CXX) -c -o $@ $< $(GPU_FLAG) $(DFFT_CUDA_MPI) -DSWFFT_$(DFFT_GPU) -DSWFFT_PLATFORM=$(DFFT_PLATFORM) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $(DFFT_OPENMP) -fPIC

$(DFFT_BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(DFFT_CUDA_CC) -o $@ $< $(GPU_FLAG) $(DFFT_CUDA_MPI) -DSWFFT_$(DFFT_GPU) -DSWFFT_PLATFORM=$(DFFT_PLATFORM) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $(DFFT_MPI_INCLUDE) $(DFFT_CUDA_FLAGS) $(DFFT_CUDA_ARCH) -c

$(DFFT_LIB_DIR)/$(DFFT_AR): $(OUTPUTS)
	mkdir -p $(@D)
	ar cr $@ $^
