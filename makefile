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

DFFT_CUDA_MPI ?=

DFFT_GPU ?= CUDA
FFT_BACKEND ?= GPU FFTW
DIST_BACKEND ?= ALLTOALL PAIRWISE

DFFT_FFT_BACKEND_DEFINES ?= $(FFT_BACKEND:%=-D%)
DFFT_DIST_BACKEND_DEFINES ?= $(DIST_BACKEND:%=-D%)

DFFT_MPI_CC ?= mpicc -O3
DFFT_MPI_CXX ?= mpicxx -O3
DFFT_CUDA_CC ?= nvcc -O3

SOURCEDIR ?= src
SOURCES := $(shell find $(SOURCEDIR) -name '*.cpp') $(shell find $(SOURCEDIR) -name '*.cu')
OBJECTS := $(SOURCES:%.cpp=%.o)
OBJECTS := $(OBJECTS:%.cu=%.o)
OUTPUTS := $(OBJECTS:src%=lib%)

#$(patsubst .cpp,.o,$(wildcard src/**/*.cpp) $(wildcard src/*.cpp))

main: $(DFFT_AR)

.PHONY: clean
clean:
	rm -rf $(DFFT_BUILD_DIR)
	rm -rf $(DFFT_LIB_DIR)

$(DFFT_LIB_DIR):
	mkdir -p $(DFFT_LIB_DIR)
	mkdir -p $(DFFT_FFT_LIB_DIR)
	mkdir -p $(DFFT_ALLTOALL_LIB_DIR)
	mkdir -p $(DFFT_PAIRWISE_LIB_DIR)

$(DFFT_LIB_DIR)/%.o: src/%.cpp | $(DFFT_LIB_DIR)
	$(DFFT_MPI_CXX) $(DFFT_CUDA_MPI) -D$(DFFT_GPU) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) -c -o $@ $<

$(DFFT_LIB_DIR)/%.o: src/%.cu | $(DFFT_LIB_DIR)
	$(DFFT_CUDA_CC) $(DFFT_CUDA_MPI) -D$(DFFT_GPU) $(DFFT_DIST_BACKEND_DEFINES) $(DFFT_FFT_BACKEND_DEFINES) $(DFFT_INCLUDE) $(DFFT_CUDA_FLAGS) $(DFFT_CUDA_ARCH) -c -o $@ $<

$(DFFT_AR): $(OUTPUTS) | $(DFFT_LIB_DIR)
	ar cr $@ $^
	