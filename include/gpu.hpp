/**
 * @file gpu.hpp
 * @brief Header file for GPU-related operations and types in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_GPU_HPP_
#define _SWFFT_GPU_HPP_

#ifdef SWFFT_GPU
#ifdef SWFFT_CUDA
#include <cuda_runtime.h>
#include <cufft.h>

namespace SWFFT {
/**
 * @typedef complexDoubleDevice
 * @brief Type alias for double-precision complex numbers on GPU using CUDA.
 */
typedef cufftDoubleComplex complexDoubleDevice;

/**
 * @typedef complexFloatDevice
 * @brief Type alias for single-precision complex numbers on GPU using CUDA.
 */
typedef cufftComplex complexFloatDevice;
} // namespace SWFFT

#define gpufftHandle cufftHandle

#define gpufftPlan1d cufftPlan1d

#define gpufftPlan3d cufftPlan3d

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

#define gpuLaunch(kernel, numBlocks, blockSize, ...)                           \
    kernel<<<numBlocks, blockSize>>>(__VA_ARGS__)

#define gpuMemcpyAsync cudaMemcpyAsync

#define gpuEventCreate cudaEventCreate

#else // SWFFT_CUDA

#ifdef SWFFT_HIP

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

#define gpufftPlan3d hipfftPlan3d

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

#define gpuLaunch(kernel, numBlocks, blockSize, ...)                           \
    kernel<<<dim3(numBlocks), dim3(blockSize), 0, 0>>>(__VA_ARGS__)

#endif // SWFFT_HIP

#endif // ~SWFFT_CUDA

namespace SWFFT {

/**
 * @brief Allocates memory for an array of complexDoubleDevice structures on the
 * GPU.
 *
 * @param ptr A pointer to the pointer that will hold the address of the
 * allocated memory.
 * @param sz The size of the memory to allocate.
 */
inline void swfftAlloc(complexDoubleDevice** ptr, size_t sz) {
    gpuMalloc(ptr, sz);
}

/**
 * @brief Allocates memory for an array of complexFloatDevice structures on the
 * GPU.
 *
 * @param ptr A pointer to the pointer that will hold the address of the
 * allocated memory.
 * @param sz The size of the memory to allocate.
 */
inline void swfftAlloc(complexFloatDevice** ptr, size_t sz) {
    gpuMalloc(ptr, sz);
}

/**
 * @brief Frees the memory allocated for an array of complexDoubleDevice
 * structures on the GPU.
 *
 * @param ptr A pointer to the memory to free.
 */
inline void swfftFree(complexDoubleDevice* ptr) { gpuFree(ptr); }

/**
 * @brief Frees the memory allocated for an array of complexFloatDevice
 * structures on the GPU.
 *
 * @param ptr A pointer to the memory to free.
 */
inline void swfftFree(complexFloatDevice* ptr) { gpuFree(ptr); }
} // namespace SWFFT

#else  // SWFFT_GPU
/**
 * @struct int3
 * @brief A structure representing a 3-dimensional integer vector.
 */
struct int3 {
    int x;
    int y;
    int z;
};

/**
 * @brief Creates an int3 structure.
 *
 * @param x X component of the vector.
 * @param y Y component of the vector.
 * @param z Z component of the vector.
 * @return int3 A structure representing a 3-dimensional integer vector.
 */
inline int3 make_int3(int x, int y, int z) {
    int3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}
#endif // ~SWFFT_GPU

#endif // _SWFFT_GPU_HPP_
