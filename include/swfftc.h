#define _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,Func) SWFFT ## __ ## DistBackend ## _ ## MPI_T ## _ ## FFTBackend ## __ ## Func
#define _GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend) SWFFT ## __ ## DistBackend ## _ ## MPI_T ## _ ## FFTBackend

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct{
  double max;
  double min;
  double sum;
  double avg;
  double var;
  double stdev;
} timing_stats_t;

#ifdef SWFFT_GPU

typedef struct{
    double x;
    double y;
} complexDoubleDevice;

#endif

typedef struct{
    double x;
    double y;
} complexDoubleHost;

#ifdef SWFFT_GPU

typedef struct{
    float x;
    float y;
} complexFloatDevice;

#endif

typedef struct{
    float x;
    float y;
} complexFloatHost;

#define GENERATE(DistBackend,MPI_T,FFTBackend)\
typedef struct { } _GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend);\
_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,new)(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, int ks_as_block);\
_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,del)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
timing_stats_t _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,printLastTime)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
timing_stats_t _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,getLastTime)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,set_nsends)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int x);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,set_delegate)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int x);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,synchronize)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,test_distribution)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_ksx)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int idx);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_ksy)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int idx);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_ksz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int idx);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_rsx)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int idx);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_rsy)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int idx);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_rsz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, int idx);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,ngx)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,ngy)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,ngz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,global_size)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_ngx)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_ngy)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_ngz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_size)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,buff_sz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,coordsx)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,coordsy)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,coordsz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,dimsx)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,dimsy)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,dimsz)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,rank)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
MPI_Comm _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,comm)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,world_size)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,world_rank)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,query)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft);

#define GENERATE_FFTS(DistBackend,MPI_T,FFTBackend,T)\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,forward ## _ ## T ## _ ## T)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, T* data, T* scratch);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,forward ## _ ## T)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, T* data);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,backward ## _ ## T ## _ ## T)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, T* data, T* scratch);\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,backward ## _ ## T)(_GENERATE_STRUCT(DistBackend,MPI_T,FFTBackend)* my_swfft, T* data);

#ifdef SWFFT_GPU
#define GENERATE_MPI_PAIRWISE(FFTBackend)\
GENERATE(Pairwise,CPUMPI,FFTBackend)\
GENERATE_FFTS(Pairwise,CPUMPI,FFTBackend,complexDoubleDevice)\
GENERATE_FFTS(Pairwise,CPUMPI,FFTBackend,complexFloatDevice)\
GENERATE_FFTS(Pairwise,CPUMPI,FFTBackend,complexDoubleHost)\
GENERATE_FFTS(Pairwise,CPUMPI,FFTBackend,complexFloatHost)

#ifdef SWFFT_NOCUDAMPI

#define GENERATE_MPI(DistBackend,FFTBackend) \
GENERATE(DistBackend,CPUMPI,FFTBackend)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexDoubleDevice)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexFloatDevice)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexDoubleHost)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexFloatHost)

#else

#define GENERATE_MPI(DistBackend,FFTBackend) \
GENERATE(DistBackend,CPUMPI,FFTBackend)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexDoubleDevice)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexFloatDevice)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexDoubleHost)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexFloatHost)\
GENERATE(DistBackend,GPUMPI,FFTBackend)\
GENERATE_FFTS(DistBackend,GPUMPI,FFTBackend,complexDoubleDevice)\
GENERATE_FFTS(DistBackend,GPUMPI,FFTBackend,complexFloatDevice)\
GENERATE_FFTS(DistBackend,GPUMPI,FFTBackend,complexDoubleHost)\
GENERATE_FFTS(DistBackend,GPUMPI,FFTBackend,complexFloatHost)

#endif
#else

#define GENERATE_MPI(DistBackend,FFTBackend) \
GENERATE(DistBackend,CPUMPI,FFTBackend)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexDoubleHost)\
GENERATE_FFTS(DistBackend,CPUMPI,FFTBackend,complexFloatHost)

#define GENERATE_MPI_PAIRWISE(FFTBackend)\
GENERATE(Pairwise,CPUMPI,FFTBackend)\
GENERATE_FFTS(Pairwise,CPUMPI,FFTBackend,complexDoubleHost)\
GENERATE_FFTS(Pairwise,CPUMPI,FFTBackend,complexFloatHost)

#endif

#if defined(SWFFT_GPU) && defined(SWFFT_CUFFT)

#define GENERATE_CUFFT(DistBackend)\
GENERATE_MPI(DistBackend,gpuFFT)

#define GENERATE_CUFFT_PAIRWISE()\
GENERATE_MPI_PAIRWISE(gpuFFT)

#else

#define GENERATE_CUFFT(DistBackend)
#define GENERATE_CUFFT_PAIRWISE()

#endif

#if defined(SWFFT_FFTW)

#define GENERATE_FFTW(DistBackend)\
GENERATE_MPI(DistBackend,fftw);

#define GENERATE_FFTW_PAIRWISE()\
GENERATE_MPI_PAIRWISE(fftw)

#else

#define GENERATE_FFTW(DistBackend)

#define GENERATE_FFTW_PAIRWISE()

#endif

#ifdef SWFFT_PAIRWISE

GENERATE_CUFFT_PAIRWISE()

GENERATE_FFTW_PAIRWISE()

#endif

#ifdef SWFFT_ALLTOALL

#ifdef SWFFT_GPU

GENERATE_CUFFT(AllToAllGPU)
GENERATE_FFTW(AllToAllGPU)

#endif

GENERATE_CUFFT(AllToAllCPU)
GENERATE_FFTW(AllToAllCPU)

#endif

#ifdef SWFFT_HQFFT

#ifdef SWFFT_GPU

GENERATE_CUFFT(HQA2AGPU)

GENERATE_CUFFT(HQPairGPU)

#endif

GENERATE_CUFFT(HQA2ACPU)

GENERATE_CUFFT(HQPairCPU)

#endif

#ifdef SWFFT_GPU
#ifdef SWFFT_GPUDELEGATE

GENERATE_CUFFT(GPUDelegate)

#endif
#endif

void swfftAlloc_complexDoubleHost(complexDoubleHost** ptr, size_t sz);

void swfftAlloc_complexFloatHost(complexFloatHost** ptr, size_t sz);

void swfftFree_complexDoubleHost(complexDoubleHost* ptr);

void swfftFree_complexFloatHost(complexFloatHost* ptr);

#ifdef SWFFT_GPU
void swfftAlloc_complexDoubleDevice(complexDoubleDevice** ptr, size_t sz);

void swfftAlloc_complexFloatDevice(complexFloatDevice** ptr, size_t sz);

void swfftFree_complexDoubleDevice(complexDoubleDevice* ptr);

void swfftFree_complexFloatDevice(complexFloatDevice* ptr);
#endif