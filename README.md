# SWFFT-GPU

SWFFT is a scalable, high-performance 3D Fast Fourier Transform (FFT) library designed for distributed-memory parallel systems. It supports CUDA-Aware MPI and uses FFTW and cuFFT as FFT backends.

### Features
* CPU and GPU FFT Backends: Supports both FFTW (CPU) and cuFFT (GPU), and can easily be modified to use other backends.
* GPU Acceleration: CUDA-Aware MPI and GPU accelerated FFTs and reordering operations.
* Multiple Communication Strategies: Various types of All-to-All and Pairwise communication methods.

## Building

### Prerequesites
* C++ Compiler (C++11 or later)
* CUDA Compiler (unless building for CPU only mode)
* MPI (tested with MPICH and OpenMPI, CUDA-Aware MPI optional)
* One (or both) of
    * fftw
    * cuFFT (hipFFT support is untested)

### Building with `make`

Use the `makefile` to build the library and tests. This will create two directories, `lib` and `build`. Tests/benchmarks are placed in `build`, and `libswfft.a` is placed in `lib`. Recommended usage: link against `libswfft.a` and include `swfft.hpp`.

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

```
make USE_CUDAMPI=TRUE
```

will build using cuda-aware mpi.

#### All options

All the options are as follows

```
make FFT_BACKEND="FFTW CUFFT" DIST_BACKEND="ALLTOALL PAIRWISE HQFFT GPUDELEGATE" USE_GPU=TRUE|FALSE (default = TRUE) USE_OMP=TRUE|FALSE (default = FALSE) USE_CUDAMPI=TRUE|FALSE (default = False)
```

#### Make Variables

The Make variables used by the build system and their defaults are as follows:

```makefile
CUDA_PATH ?= /usr/local/cuda

# This is only used for certain tests.
# If you are only compiling build/testdfft, you can set this to blank.
DFFT_MPI_INCLUDE ?= -I/usr/include/x86_64-linux-gnu/mpich

DFFT_CUDA_ARCH ?= -gencode=arch=compute_60,code=sm_60

DFFT_FFTW_HOME ?= $(shell dirname $(shell dirname $(shell which fftw-wisdom)))

DFFT_MPI_CC ?= mpicc -O3

DFFT_MPI_CXX ?= mpicxx -O3

DFFT_CUDA_CC ?= nvcc -O3
```

### Building with `cmake`
Coming soon (maybe...)

## Usage

### Configuring SWFFT headers
```c++
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SWFFT_GPU
#define SWFFT_CUFFT
#define SWFFT_ALLTOALL

#include "swfft.hpp"

using namespace SWFFT;
```

### Initializing CPU threads
```c++
swfft_init_threads(4);
```

### Creating a SWFFT instance
```c++
swfft<AllToAllGPU,CPUMPI,gpuFFT> my_swfft(MPI_COMM_WORLD,256,256,256,64);
```

### Querying the SWFFT configuration
```c++
my_swfft.query(); // prints information
```

### Allocating/Freeing SWFFT buffers
```c++
complexDoubleHost* data;
swfftAlloc(&data, sizeof(complexDoubleHost) * my_swfft.buff_sz());
complexDoubleHost* scratch;
swfftAlloc(&scratch, sizeof(complexDoubleHost) * my_swfft.buff_sz());

// ...

swfftFree(data);
swfftFree(scratch);
```

### Performing FFT Operations
```c++
my_swfft.forward(data,scratch);
my_swfft.backward(data,scratch);
```

### Querying index position in k-space/real-space
```c++
int3 ks = my_swfft.get_ks(0);
int3 rs = my_swfft.get_rs(0);
```

## Minimal example
```c++
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SWFFT_GPU
#define SWFFT_CUFFT
#define SWFFT_ALLTOALL

#include "swfft.hpp"

using namespace SWFFT;

int test(){
    swfft<AllToAllGPU,CPUMPI,gpuFFT> my_swfft(MPI_COMM_WORLD,256,256,256,64);
    my_swfft.query();

    complexDoubleDevice* data; swfftAlloc(&data,sizeof(complexDoubleDevice) * my_swfft.buff_sz());
    complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * my_swfft.buff_sz());

    int3 this_position_in_kspace = my_swfft.get_ks(0);
    printf("for rank %d, index 0 is at (%d %d %d) in kspace\n",my_swfft.rank(),this_position_in_kspace.x,this_position_in_kspace.y,this_position_in_kspace.z);

    my_swfft.forward(data,scratch);
    my_swfft.backward(data,scratch);

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

## Tests/Benchmarks

Running `mpirun -n <n> build/testdfft <ngx> [ngy ngz]` will test all possible configurations of swfft that were compiled.

## Interface

### Macros
```
SWFFT_GPU

SWFFT_ALLTOALL
SWFFT_PAIRWISE
SWFFT_HQFFT

SWFFT_FFTW
SWFFT_CUFFT
```

### Backend Options
```
DistBackend = SWFFT::AllToAllCPU | SWFFT::AllToAllGPU | SWFFT::Pairwise | SWFFT::HQPairGPU | SWFFT::HQA2AGPU | SWFFT::HQPairCPU | SWFFT::HQA2ACPU

MPI_T = SWFFT::CPUMPI | SWFFT::GPUMPI

FFTBackend = SWFFT::gpuFFT | SWFFT::fftw
```

### Constructors
```c++
SWFFT::swfft<DistBackend,MPI_T,FFTBackend> my_swfft(MPI_Comm comm, int ngx, int blockSize = 64, bool ks_as_block = true);
```
or
```
SWFFT::swfft<DistBackend,MPI_T,FFTBackend> my_swfft(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize = 64, bool ks_as_block = true);
```

### Accessors
```c++
int3 my_swfft.get_ks(int idx);

int3 my_swfft.get_rs(int idx);

int my_swfft.ngx();

int my_swfft.ngy();

int my_swfft.ngz();

int3 my_swfft.ng();

int my_swfft.ng(int direction);

int my_swfft.local_ngx();

int my_swfft.local_ngy();

int my_swfft.local_ngz();

int3 my_swfft.local_ng();

int my_swfft.local_ng(int direction);

int3 my_swfft.coords();

int3 my_swfft.dims();

MPI_Comm my_swfft.comm();

int my_swfft.rank();

int my_swfft.world_rank();

int my_swfft.world_size();

int my_swfft.global_size();

int my_swfft.local_size();

size_t my_swfft.buff_sz();

void my_swfft.query(); // prints useful information
```

### FFT
```c++
datatype = SWFFT::complexDoubleDevice | SWFFT::complexFloatDevice | SWFFT::complexDoubleHost | SWFFT::complexFloatHost

void my_swfft.forward(datatype* data, datatype* scratch);

void my_swfft.backward(datatype* data, datatype* scratch);

void my_swfft.forward(datatype* data);

void my_swfft.backward(datatype* data);
```

### Timings
```c++
struct SWFFT::timing_stats_t{
    double max;
    double min;
    double sum;
    double avg;
    double var;
    double stdev;
}

timing_stats_t my_swfft.printLastTime(); // prints the time for the last FFT

timing_stats_t my_swfft.getLastTime();
```

### Threading
```c++
void swfft_init_threads(int nthreads = 0);
```