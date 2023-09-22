DFFT_BUILD_DIR ?= build
DFFT_LIB_DIR ?= lib

DFFT_PAIRWISE_LIB_DIR ?= $(DFFT_LIB_DIR)/pairwise
DFFT_ALLTOALL_LIB_DIR ?= $(DFFT_LIB_DIR)/alltoall
DFFT_FFT_LIB_DIR ?= $(DFFT_LIB_DIR)/fftbackends

DFFT_PLATFORM ?= unknown

DFFT_AR ?= lib/libswfft.a

DFFT_CUDA_LIB ?= /usr/local/cuda/lib64
DFFT_CUDA_INC ?= /usr/local/cuda/include

DFFT_INCLUDE ?= -Iinclude -I$(DFFT_CUDA_INC)
DFFT_LD ?= -L$(DFFT_CUDA_LIB)

DFFT_CUDA_ARCH ?= -gencode=arch=compute_60,code=sm_60

DFFT_CUDA_LD ?= -lcufft -lcudart

DFFT_CUDA_FLAGS ?= -lineinfo -Xptxas -v -Xcompiler="-fPIC"

DFFT_CUDA_MPI ?= -Dnocudampi

DFFT_GPU ?= CUDA
USE_GPU ?= TRUE
ifeq ($(USE_GPU), TRUE)
GPU_FLAG = -DSWFFT_GPU
else
GPU_FLAG =
endif
FFT_BACKEND ?= CUFFT FFTW
DIST_BACKEND ?= ALLTOALL PAIRWISE

USE_OMP ?= TRUE

DFFT_FFTW_HOME ?= $(shell dirname $(shell dirname $(shell which fftw-wisdom)))
DFFT_FFTW_CPPFLAGS ?= -I$(DFFT_FFTW_HOME)/include
ifeq ($(USE_OMP), TRUE)
DFFT_FFTW_LDFLAGS ?= -L$(DFFT_FFTW_HOME)/lib -lfftw3_omp -lfftw3 -lfftw3f -lm
DFFT_OPENMP ?= -fopenmp
else
DFFT_FFTW_LDFLAGS ?= -L$(DFFT_FFTW_HOME)/lib -lfftw3 -lfftw3f -lm
DFFT_OPENMP ?=
endif

DFFT_FFT_BACKEND_DEFINES ?= $(FFT_BACKEND:%=-DSWFFT_%)
DFFT_DIST_BACKEND_DEFINES ?= $(DIST_BACKEND:%=-DSWFFT_%)

DFFT_MPI_CC ?= mpicc -O3
DFFT_MPI_CXX ?= mpicxx -O3
DFFT_CUDA_CC ?= nvcc -O3

SOURCEDIR ?= src
ifeq ($(USE_GPU), TRUE)
SOURCES := $(shell find $(SOURCEDIR) -name '*.cpp') $(shell find $(SOURCEDIR) -name '*.cu')
OBJECTS := $(SOURCES:%.cpp=%.o)
OBJECTS := $(OBJECTS:%.cu=%.o)
OUTPUTS := $(OBJECTS:src%=lib%)
else
SOURCES := $(shell find $(SOURCEDIR) -name '*.cpp')
OBJECTS := $(SOURCES:%.cpp=%.o)
OUTPUTS := $(OBJECTS:src%=lib%)
endif


#$(patsubst .cpp,.o,$(wildcard src/**/*.cpp) $(wildcard src/*.cpp))

main: $(DFFT_BUILD_DIR)/testdfft

#$(DFFT_BUILD_DIR)/%: test/%.cpp | $(DFFT_AR) $(DFFT_BUILD_DIR)
#	$(DFFT_MPI_CXX) $(GPU_FLAG) $(DFFT_CUDA_MPI) -D$(DFFT_GPU) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $< $(DFFT_FFTW_LDFLAGS) -L$(DFFT_LIB_DIR) -lswfft -L$(DFFT_CUDA_LIB) $(DFFT_CUDA_LD) -o $@

$(DFFT_BUILD_DIR)/%: test/%.cpp $(DFFT_AR) | $(DFFT_BUILD_DIR)
	$(DFFT_MPI_CXX) $(GPU_FLAG) $(DFFT_CUDA_MPI) -DSWFFT_$(DFFT_GPU) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $^ $(DFFT_FFTW_LDFLAGS) $(DFFT_OPENMP) -L$(DFFT_CUDA_LIB) $(DFFT_CUDA_LD) -o $@

.PHONY: clean
clean:
	rm -rf $(DFFT_BUILD_DIR)
	rm -rf $(DFFT_LIB_DIR)

$(DFFT_LIB_DIR):
	mkdir -p $(DFFT_LIB_DIR)
	mkdir -p $(DFFT_FFT_LIB_DIR)
	mkdir -p $(DFFT_ALLTOALL_LIB_DIR)
	mkdir -p $(DFFT_PAIRWISE_LIB_DIR)

$(DFFT_BUILD_DIR):
	mkdir -p $(DFFT_BUILD_DIR)

$(DFFT_LIB_DIR)/%.o: src/%.cpp | $(DFFT_LIB_DIR)
	$(DFFT_MPI_CXX) $(GPU_FLAG) $(DFFT_CUDA_MPI) -DSWFFT_$(DFFT_GPU) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $(DFFT_OPENMP) -c -o $@ $<

$(DFFT_LIB_DIR)/%.o: src/%.cu | $(DFFT_LIB_DIR)
	$(DFFT_CUDA_CC) $(GPU_FLAG) $(DFFT_CUDA_MPI) -DSWFFT_$(DFFT_GPU) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $(DFFT_CUDA_FLAGS) $(DFFT_CUDA_ARCH) -c -o $@ $<

$(DFFT_AR): $(OUTPUTS) | $(DFFT_LIB_DIR)
	ar cr $@ $^
	