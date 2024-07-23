/**
 * @file testfft_backend.hpp
 * @brief Header file for TestFFT class in the SWFFT namespace.
 */

#ifndef _SWFFT_TESTFFT_BACKEND_HPP_
#define _SWFFT_TESTFFT_BACKEND_HPP_

#include "complex-type.hpp"
#include "fftbackend.hpp"
#include "gpu.hpp"
#include "query.hpp"

namespace SWFFT {
/**
 * @class TestFFT
 * @brief A test implementation of the FFTBackend_T class for verifying FFT
 * operations.
 *
 * This class provides a simple pass-through implementation of the FFT
 * operations, copying data from input buffer to output buffer.
 */
class TestFFT : public FFTBackend_T {
  private:
    /**
     * @brief Internal CPU function to perform a simple copy operation.
     *
     * @tparam T Data type of the buffer elements.
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    template <class T> inline void _cpu(T* buff1, T* buff2, int ng, int nFFTs) {
        for (int i = 0; i < ng * nFFTs; i++) {
            buff2[i] = buff1[i];
        }
    }
#ifdef SWFFT_GPU
    /**
     * @brief Internal GPU function to perform a simple copy operation using GPU
     * memory.
     *
     * @tparam T Data type of the buffer elements.
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    template <class T> inline void _gpu(T* buff1, T* buff2, int ng, int nFFTs) {
        gpuMemcpy(buff2, buff1, sizeof(T) * ng * nFFTs,
                  gpuMemcpyDeviceToDevice);
    }
#endif

  public:
    /**
     * @brief Default constructor for TestFFT.
     */
    TestFFT(){};

    /**
     * @brief Destructor.
     */
    ~TestFFT(){};

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on GPU with double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int ng,
                 int nFFTs) {
        _gpu(buff1, buff2, ng, nFFTs);
    }

    /**
     * @brief Perform backward FFT on GPU with double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                  int ng, int nFFTs) {
        _gpu(buff1, buff2, ng, nFFTs);
    }

    /**
     * @brief Perform forward FFT on GPU with single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexFloatDevice* buff1, complexFloatDevice* buff2, int ng,
                 int nFFTs) {
        _gpu(buff1, buff2, ng, nFFTs);
    }

    /**
     * @brief Perform backward FFT on GPU with single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexFloatDevice* buff1, complexFloatDevice* buff2, int ng,
                  int nFFTs) {
        _gpu(buff1, buff2, ng, nFFTs);
    }
#endif
    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexDoubleHost* buff1, complexDoubleHost* buff2, int ng,
                 int nFFTs) {
        _cpu(buff1, buff2, ng, nFFTs);
    }

    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexDoubleHost* buff1, complexDoubleHost* buff2, int ng,
                  int nFFTs) {
        _cpu(buff1, buff2, ng, nFFTs);
    }

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexFloatHost* buff1, complexFloatHost* buff2, int ng,
                 int nFFTs) {
        _cpu(buff1, buff2, ng, nFFTs);
    }

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param buff1 Pointer to the input buffer.
     * @param buff2 Pointer to the output buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexFloatHost* buff1, complexFloatHost* buff2, int ng,
                  int nFFTs) {
        _cpu(buff1, buff2, ng, nFFTs);
    }
};

/**
 * @brief Query the name of the TestFFT class.
 *
 * @return const char* The name of the TestFFT class.
 */
template <> inline const char* queryName<TestFFT>() { return "TestFFT"; }
} // namespace SWFFT

#endif