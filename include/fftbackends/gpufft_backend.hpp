/**
 * @file gpufft_backend.hpp
 * @brief Header file for GPU FFT backend classes and functions in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_GPUFFT_BACKEND_HPP_
#define _SWFFT_GPUFFT_BACKEND_HPP_

#ifdef SWFFT_CUFFT
#ifdef SWFFT_GPU

#define N_FFT_CACHE 100

#include "complex-type.hpp"
#include "gpu.hpp"
#include "query.hpp"

#include "fftbackend.hpp"

namespace SWFFT {
/**
 * @enum fftdirection
 * @brief Enum to define FFT directions.
 */
enum fftdirection { FFT_FORWARD, FFT_BACKWARD };

/**
 * @class GPUPlanWrapper
 * @brief Class to wrap GPU FFT plan details.
 */
class GPUPlanWrapper {
  public:
    int nFFTs;         /**< Number of FFTs */
    gpufftHandle plan; /**< GPU FFT plan handle */
    gpufftType t;      /**< GPU FFT type */
    int ng;            /**< Number of grid cells */
    bool valid;        /**< Plan validity flag */
};

/**
 * @class GPUPlanManager
 * @brief Class to manage GPU FFT plans and perform FFT operations.
 */
class GPUPlanManager : public FFTBackend_T {
  public:
    GPUPlanWrapper plans[N_FFT_CACHE]; /**< Array of cached GPU FFT plans */

    /**
     * @brief Constructor for GPUPlanManager.
     */
    GPUPlanManager();

    /**
     * @brief Destructor for GPUPlanManager.
     */
    ~GPUPlanManager();

    /**
     * @brief Query the GPU FFT backend.
     */
    void query();

    /**
     * @brief Find or create a GPU FFT plan.
     *
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs.
     * @param t GPU FFT type.
     * @return gpufftHandle Handle to the GPU FFT plan.
     */
    gpufftHandle find_plan(int ng, int nFFTs, gpufftType t);

    /**
     * @brief Perform forward FFT on GPU with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch,
                 int ng, int nFFTs);

    /**
     * @brief Perform forward FFT on GPU with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng,
                 int nFFTs);

    /**
     * @brief Perform forward FFT on on GPU with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexDoubleHost* data, complexDoubleHost* scratch, int ng,
                 int nFFTs);

    /**
     * @brief Perform forward FFT on GPU with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexFloatHost* data, complexFloatHost* scratch, int ng,
                 int nFFTs);

    /**
     * @brief Perform backward FFT on GPU with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch,
                  int ng, int nFFTs);

    /**
     * @brief Perform backward FFT on GPU with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexFloatDevice* data, complexFloatDevice* scratch, int ng,
                  int nFFTs);

    /**
     * @brief Perform backward FFT on GPU with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexDoubleHost* data, complexDoubleHost* scratch, int ng,
                  int nFFTs);

    /**
     * @brief Perform backward FFT on GPU with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexFloatHost* data, complexFloatHost* scratch, int ng,
                  int nFFTs);
};

/**
 * @typedef gpuFFT
 * @brief Typedef for GPUPlanManager.
 */
typedef GPUPlanManager gpuFFT;

/**
 * @brief Query the name of the gpuFFT class.
 *
 * @return const char* The name of the gpuFFT class.
 */
template <> inline const char* queryName<gpuFFT>() { return "gpuFFT"; }

#endif // SWFFT_GPU
#endif // SWFFT_CUFFT

} // namespace SWFFT

#endif // _SWFFT_GPUFFT_BACKEND_HPP_