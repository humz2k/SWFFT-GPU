# SWFFT-GPU (WIP)

## Building

### System Requirements

**Required**

* MPI (gpu-aware MPI optional)

**Optional**

* fftw (double and single precision)
* cuFFT/hipFFT

### Makefiles

Use the `makefile` to build the library and tests. This will create two directories, `lib` and `build`. Tests/benchmarks are placed in `build`, and `libswfft.a` is placed in `lib`.

#### Examples

```
make
```

will build a MPI and GPU version of the code.


```
make USE_OMP=TRUE
```

will build a MPI/OpenMP and GPU version of the code.


```
make USE_GPU=FALSE USE_OMP=TRUE
```

will build a MPI/OpenMP version of the code.


```
make FFT_BACKEND="FFTW" DIST_BACKEND="ALLTOALL PAIRWISE" USE_GPU=FALSE
```

will build with only `fftw` for the FFT backend, and `AllToAllCPU`/`Pairwise` for the distribution backend.

#### All options

All the options are as follows

```
make FFT_BACKEND="FFTW CUFFT" DIST_BACKEND="ALLTOALL PAIRWISE HQFFT GPUDELEGATE" USE_GPU=TRUE|FALSE USE_OMP=TRUE|FALSE
```

### Environment Variables

The environment variables used by the build system and their defaults are as follows:

```
DFFT_CUDA_LIB ?= /usr/local/cuda/lib64

DFFT_CUDA_INC ?= /usr/local/cuda/include

DFFT_CUDA_ARCH ?= -gencode=arch=compute_60,code=sm_60

DFFT_FFTW_HOME ?= $(shell dirname $(shell dirname $(shell which fftw-wisdom)))

DFFT_MPI_CC ?= mpicc -O3

DFFT_MPI_CXX ?= mpicxx -O3

DFFT_CUDA_CC ?= nvcc -O3
```

# Tests/Benchmarks

Running `build/testdfft <ngx> [ngy ngz]` will test all possible configurations of swfft that were compiled.

# Interface



# Minimal example

```
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "swfft.hpp"

int test(){
    swfft<AllToAllGPU,CPUMPI,gpuFFT> my_swfft(MPI_COMM_WORLD,256,256,256,64);
    my_swfft.query();

    complexDoubleDevice* data; swfftAlloc(&data,sizeof(complexDoubleDevice) * my_swfft.buff_sz());
    complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * my_swfft.buff_sz());

    int3 this_position_in_kspace = my_swfft.get_ks(0);
    printf("index 0 is at (%d %d %d) in kspace\n",this_position_in_kspace.x,this_position_in_kspace.y,this_position_in_kspace.z);

    my_swfft.forward(data,scratch);
    my_swfft.synchronize(); //only important if using the GPUDelegate backend!
    my_swfft.backward(data,scratch);
    my_swfft.synchronize();

    swfftFree(data);
    swfftFree(scratch);
}

int main(){

    MPI_Init(NULL,NULL);

    test();

    MPI_Finalize();

    return 0;
}
```