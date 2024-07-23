/**
 * @file copy_buffers.hpp
 * @brief Header file for buffer copy classes in the SWFFT namespace.
 */

#ifndef _SWFFT_COPY_BUFFERS_HPP_
#define _SWFFT_COPY_BUFFERS_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"
#include <stdio.h>

namespace SWFFT {
/**
 * @class copyBuffersBase
 * @brief Base class for buffer copy operations.
 *
 * @tparam T Data type of the buffer elements.
 */
template <class T> class copyBuffersBase {
  protected:
    T* dest; /**< Pointer to the destination buffer */
    T* src;  /**< Pointer to the source buffer */
    int n;   /**< Number of elements to copy */
  public:
    /**
     * @brief Constructor for copyBuffersBase.
     *
     * @param dest_ Pointer to the destination buffer.
     * @param src_ Pointer to the source buffer.
     * @param n_ Number of elements to copy.
     */
    copyBuffersBase(T* dest_, T* src_, int n_)
        : dest(dest_), src(src_), n(n_) {}

    /**
     * @brief Virtual function to wait for the copy operation to complete.
     */
    void wait() {}
};

/**
 * @class copyBuffersCPU
 * @brief Class for CPU buffer copy operations.
 *
 * @tparam T Data type of the buffer elements.
 */
template <class T> class copyBuffersCPU : public copyBuffersBase<T> {
  public:
    /**
     * @brief Constructor for copyBuffersCPU.
     *
     * This constructor performs the copy operation on the CPU.
     *
     * @param dest_ Pointer to the destination buffer.
     * @param src_ Pointer to the source buffer.
     * @param n_ Number of elements to copy.
     */
    copyBuffersCPU(T* dest_, T* src_, int n_)
        : copyBuffersBase<T>(dest_, src_, n_) {
        for (int i = 0; i < this->n; i++) {
            this->dest[i] = this->src[i];
        }
    }
};

/**
 * @class copyBuffers
 * @brief Default copyBuffers class, aliasing to copyBuffersCPU.
 *
 * @tparam T Data type of the buffer elements.
 */
template <class T> class copyBuffers : public copyBuffersCPU<T> {
  public:
    using copyBuffersCPU<T>::copyBuffersCPU;
};

#ifdef SWFFT_GPU
/**
 * @class copyBuffersGPU
 * @brief Class for GPU buffer copy operations.
 *
 * @tparam T Data type of the buffer elements.
 */
template <class T> class copyBuffersGPU : public copyBuffersBase<T> {
  private:
    gpuEvent_t event; /**< GPU event for synchronization */
  public:
    /**
     * @brief Constructor for copyBuffersGPU.
     *
     * This constructor performs the copy operation on the GPU.
     *
     * @param dest_ Pointer to the destination buffer.
     * @param src_ Pointer to the source buffer.
     * @param n_ Number of elements to copy.
     */
    copyBuffersGPU(T* dest_, T* src_, int n_)
        : copyBuffersBase<T>(dest_, src_, n_) {
        gpuEventCreate(&event);
        gpuMemcpyAsync(this->dest, this->src, this->n * sizeof(T),
                       gpuMemcpyDeviceToDevice);
        gpuEventRecord(event);
    }

    /**
     * @brief Wait for the GPU copy operation to complete.
     */
    void wait() {
        gpuEventSynchronize(event);
        gpuEventDestroy(event);
    }
};

/**
 * @brief Specialization of copyBuffers for complexDoubleDevice on the GPU.
 */
template <>
class copyBuffers<complexDoubleDevice>
    : public copyBuffersGPU<complexDoubleDevice> {
  public:
    using copyBuffersGPU::copyBuffersGPU;
};

/**
 * @brief Specialization of copyBuffers for complexFloatDevice on the GPU.
 */
template <>
class copyBuffers<complexFloatDevice>
    : public copyBuffersGPU<complexFloatDevice> {
  public:
    using copyBuffersGPU::copyBuffersGPU;
};
#endif // SWFFT_GPU

} // namespace SWFFT

#endif // _SWFFT_COPY_BUFFERS_HPP_