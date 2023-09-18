#ifndef GPUSEEN
#define GPUSEEN

#ifdef CUDA
#include <cuda_runtime.h>
#include <cufft.h>

#define complexDoubleDevice cufftDoubleComplex
#define complexFloatDevice cufftComplex
//#define complexFFT_t cufftDoubleComplex

#define gpufftHandle cufftHandle

#define gpufftPlan1d cufftPlan1d

#define gpufftDestroy cufftDestroy

#define gpufftType cufftType

#define GPUFFT_Z2Z CUFFT_Z2Z
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE

#define gpufftExecZ2Z cufftExecZ2Z
#define gpufftExecC2C cufftExecC2C

#define gpuEvent_t cudaEvent_t

#define gpuStream_t cudaStream_t

#define gpuEventDestroy cudaEventDestroy

#define gpuStreamDestroy cudaStreamDestroy

#define gpuEventRecord cudaEventRecord

#define gpuEventSynchronize cudaEventSynchronize

#define gpuMalloc cudaMalloc

#define gpuMemset cudaMemset

#define gpuMemcpy cudaMemcpy

#define gpuDeviceSynchronize cudaDeviceSynchronize

#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice

#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost

#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define gpuFree cudaFree

#define gpuStreamCreate cudaStreamCreate

#define gpuStreamSynchronize cudaStreamSynchronize

#define gpuLaunch(kernel,numBlocks,blockSize,...) kernel<<<numBlocks,blockSize>>>(__VA_ARGS__)

#define gpuMemcpyAsync cudaMemcpyAsync

#define gpuEventCreate cudaEventCreate


#else 

#ifdef HIP

#include <hip/hip_runtime_api.h>
#include <hipfft.h>

#define complexDoubleDevice hipfftDoubleComplex
#define complexFloatDevice hipfftComplex

#define GPUFFT_FORWARD HIPFFT_FORWARD
#define GPUFFT_INVERSE HIPFFT_BACKWARD
#define GPUFFT_Z2Z HIPFFT_Z2Z
#define GPUFFT_C2C HIPFFT_C2C
#define GPUFFT_SUCCESS HIPFFT_SUCCESS

#define gpufftExecZ2Z hipfftExecZ2Z
#define gpufftExecC2C hipfftExecC2C

#define gpuStream_t hipStream_t

#define gpufftHandle hipfftHandle

#define gpufftPlan1d hipfftPlan1d

#define gpufftDestroy hipfftDestroy

#define gpufftType hipfftType

#define gpuMalloc hipMalloc

#define gpuMemcpy hipMemcpy

#define gpuMemset hipMemset

#define gpuDeviceSynchronize hipDeviceSynchronize

#define gpuMemcpyHostToDevice hipMemcpyHostToDevice

#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost

#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define gpuFree hipFree

#define gpuLaunch(kernel,numBlocks,blockSize,...) kernel<<<dim3(numBlocks),dim3(blockSize),0,0>>>(__VA_ARGS__)

//#define gpuLaunch(kernel,numBlocks,blockSize,...) hipLaunchKernelGGL(kernel,dim3(numBlocks),dim3(blockSize),0,0,__VA_ARGS__)

#endif

#endif
#endif