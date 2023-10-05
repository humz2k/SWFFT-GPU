#include "swfft.hpp"

using namespace SWFFT;

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline swfft<DistBackend,MPI_T,FFTBackend>* __swfft_new(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, int ks_as_block){
    return new swfft<DistBackend,MPI_T,FFTBackend>(comm,ngx,ngy,ngz,blockSize,ks_as_block);
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline void __swfft_del(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    delete my_swfft;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline timing_stats_t __swfft_printLastTime(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->printLastTime();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline timing_stats_t __swfft_getLastTime(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->getLastTime();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline void __swfft_set_nsends(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int x){
    return my_swfft->set_nsends(x);
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline void __swfft_set_delegate(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int x){
    return my_swfft->set_delegate(x);
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline void __swfft_synchronize(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->synchronize();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_test_distribution(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->test_distribution();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_get_ksx(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){
    return my_swfft->get_ks(idx).x;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_get_ksy(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){
    return my_swfft->get_ks(idx).y;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_get_ksz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){
    return my_swfft->get_ks(idx).z;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_get_rsx(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){
    return my_swfft->get_rs(idx).x;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_get_rsy(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){
    return my_swfft->get_rs(idx).y;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_get_rsz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){
    return my_swfft->get_rs(idx).z;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_ngx(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->ngx();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_ngy(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->ngy();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_ngz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->ngz();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_global_size(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->global_size();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_local_ngx(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->local_ngx();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_local_ngy(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->local_ngy();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_local_ngz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->local_ngz();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_local_size(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->local_size();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_buff_sz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->buff_sz();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_coordsx(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->coords().x;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_coordsy(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->coords().y;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_coordsz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->coords().z;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_dimsx(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->dims().x;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_dimsy(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->dims().y;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_dimsz(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->dims().z;
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_rank(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->rank();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline MPI_Comm __swfft_comm(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->comm();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_world_size(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->world_size();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline int __swfft_world_rank(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->world_rank();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
inline void __swfft_query(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){
    return my_swfft->query();
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend, class T>
inline void __swfft_forward(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* buff1, T* buff2){
    return my_swfft->forward(buff1,buff2);
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend, class T>
inline void __swfft_forward(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* buff1){
    return my_swfft->forward(buff1);
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend, class T>
inline void __swfft_backward(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* buff1, T* buff2){
    return my_swfft->backward(buff1,buff2);
}

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend, class T>
inline void __swfft_backward(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* buff1){
    return my_swfft->backward(buff1);
}

#define _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,Func) SWFFT ## __ ## DistBackend ## _ ## MPI_T ## _ ## FFTBackend ## __ ## Func
#define GENERATE(DistBackend,MPI_T,FFTBackend) \
swfft<DistBackend,MPI_T,FFTBackend>* _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,new)(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, int ks_as_block){\
    return __swfft_new<DistBackend,MPI_T,FFTBackend>(comm,ngx,ngy,ngz,blockSize,ks_as_block);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,del)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_del<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
timing_stats_t _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,printLastTime)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_printLastTime<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
timing_stats_t _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,getLastTime)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_getLastTime<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,set_nsends)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int x){\
    return __swfft_set_nsends<DistBackend,MPI_T,FFTBackend>(my_swfft,x);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,set_delegate)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int x){\
    return __swfft_set_delegate<DistBackend,MPI_T,FFTBackend>(my_swfft,x);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,synchronize)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_synchronize<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,test_distribution)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_test_distribution<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_ksx)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){\
    return __swfft_get_ksx<DistBackend,MPI_T,FFTBackend>(my_swfft,idx);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_ksy)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){\
    return __swfft_get_ksy<DistBackend,MPI_T,FFTBackend>(my_swfft,idx);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_ksz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){\
    return __swfft_get_ksz<DistBackend,MPI_T,FFTBackend>(my_swfft,idx);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_rsx)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){\
    return __swfft_get_rsx<DistBackend,MPI_T,FFTBackend>(my_swfft,idx);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_rsy)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){\
    return __swfft_get_rsy<DistBackend,MPI_T,FFTBackend>(my_swfft,idx);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,get_rsz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, int idx){\
    return __swfft_get_rsz<DistBackend,MPI_T,FFTBackend>(my_swfft,idx);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,ngx)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_ngx<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,ngy)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_ngy<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,ngz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_ngz<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,global_size)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_global_size<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_ngx)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_local_ngx<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_ngy)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_local_ngy<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_ngz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_local_ngz<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,local_size)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_local_size<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,buff_sz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_buff_sz<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,coordsx)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_coordsx<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,coordsy)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_coordsy<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,coordsz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_coordsz<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,dimsx)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_dimsx<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,dimsy)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_dimsy<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,dimsz)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_dimsz<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,rank)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_rank<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
MPI_Comm _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,comm)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_comm<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,world_size)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_world_size<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
int _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,world_rank)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_world_rank<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,query)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft){\
    return __swfft_query<DistBackend,MPI_T,FFTBackend>(my_swfft);\
}

#define GENERATE_FFTS(DistBackend,MPI_T,FFTBackend,T)\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,forward ## _ ## T ## _ ## T)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* data, T* scratch){\
    return __swfft_forward<DistBackend,MPI_T,FFTBackend>(my_swfft,data,scratch);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,forward ## _ ## T)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* data){\
    return __swfft_forward<DistBackend,MPI_T,FFTBackend>(my_swfft,data);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,backward ## _ ## T ## _ ## T)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* data, T* scratch){\
    return __swfft_backward<DistBackend,MPI_T,FFTBackend>(my_swfft,data,scratch);\
}\
void _GENERATE_HEADER(DistBackend,MPI_T,FFTBackend,backward ## _ ## T)(swfft<DistBackend,MPI_T,FFTBackend>* my_swfft, T* data){\
    return __swfft_backward<DistBackend,MPI_T,FFTBackend>(my_swfft,data);\
}\

extern "C"{

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

void swfftAlloc_complexDoubleHost(complexDoubleHost** ptr, size_t sz){
    return swfftAlloc(ptr,sz);
}

void swfftAlloc_complexFloatHost(complexFloatHost** ptr, size_t sz){
    return swfftAlloc(ptr,sz);
}

void swfftFree_complexDoubleHost(complexDoubleHost* ptr){
    return swfftFree(ptr);
}

void swfftFree_complexFloatHost(complexFloatHost* ptr){
    return swfftFree(ptr);
}

#ifdef SWFFT_GPU
void swfftAlloc_complexDoubleDevice(complexDoubleDevice** ptr, size_t sz){
    return swfftAlloc(ptr,sz);
}

void swfftAlloc_complexFloatDevice(complexFloatDevice** ptr, size_t sz){
    return swfftAlloc(ptr,sz);
}

void swfftFree_complexDoubleDevice(complexDoubleDevice* ptr){
    return swfftFree(ptr);
}

void swfftFree_complexFloatDevice(complexFloatDevice* ptr){
    return swfftFree(ptr);
}
#endif


}