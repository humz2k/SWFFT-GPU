/**
 * @file fftbackend.hpp
 * @brief Header file for FFTBackend_T class in the SWFFT namespace.
 */

#ifndef _SWFFT_FFTBACKEND_HPP_
#define _SWFFT_FFTBACKEND_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"

namespace SWFFT {
/**
 * @class FFTBackend_T
 * @brief Abstract base class for FFT backends in the SWFFT namespace.
 *
 * This class defines the interface for FFT operations, including both forward
 * and backward transforms, for different types of complex data on both host and
 * GPU memory.
 */
class FFTBackend_T {
  public:
    /**
     * @brief Constructor for FFTBackend_T.
     */
    FFTBackend_T() {}

    /**
     * @brief Virtual destructor for FFTBackend_T.
     */
    virtual ~FFTBackend_T(){};

#ifdef SWFFT_GPU
    /**
     * @brief Pure virtual function for performing forward FFT on GPU with
     * double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void forward(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                         int ng, int nFFTs) = 0;

    /**
     * @brief Pure virtual function for performing forward FFT on GPU with
     * single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void forward(complexFloatDevice* buff1, complexFloatDevice* buff2,
                         int ng, int nFFTs) = 0;

    /**
     * @brief Pure virtual function for performing backward FFT on GPU with
     * double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void backward(complexDoubleDevice* buff1,
                          complexDoubleDevice* buff2, int ng, int nFFTs) = 0;

    /**
     * @brief Pure virtual function for performing backward FFT on GPU with
     * single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void backward(complexFloatDevice* buff1, complexFloatDevice* buff2,
                          int ng, int nFFTs) = 0;
#endif
    /**
     * @brief Pure virtual function for performing forward FFT on host with
     * double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void forward(complexDoubleHost* buff1, complexDoubleHost* buff2,
                         int ng, int nFFTs) = 0;

    /**
     * @brief Pure virtual function for performing forward FFT on host with
     * single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void forward(complexFloatHost* buff1, complexFloatHost* buff2,
                         int ng, int nFFTs) = 0;

    /**
     * @brief Pure virtual function for performing backward FFT on host with
     * double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void backward(complexDoubleHost* buff1, complexDoubleHost* buff2,
                          int ng, int nFFTs) = 0;

    /**
     * @brief Pure virtual function for performing backward FFT on host with
     * single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of groups.
     * @param nFFTs Number of FFTs to perform.
     */
    virtual void backward(complexFloatHost* buff1, complexFloatHost* buff2,
                          int ng, int nFFTs) = 0;
};
} // namespace SWFFT

#endif // _SWFFT_FFTBACKEND_HPP_