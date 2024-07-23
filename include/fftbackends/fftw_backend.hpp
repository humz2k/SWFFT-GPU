/**
 * @file fftw_backend.hpp
 * @brief Header file for FFTW backend classes and functions in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_FFTW_BACKEND_HPP_
#define _SWFFT_FFTW_BACKEND_HPP_

#define N_FFT_CACHE 100

#include "complex-type.hpp"
#include "gpu.hpp"
#include "query.hpp"

#ifdef SWFFT_FFTW

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fftw3.h>
#include <map>
#endif

#include "fftbackend.hpp"

namespace SWFFT {

#ifdef SWFFT_FFTW
/**
 * @brief Initialize FFTW threads for parallel execution.
 *
 * @param omt Number of OpenMP threads.
 * @return int Number of threads initialized, or 1 if initialization failed.
 */
static inline int swfft_fftw_init_threads(int omt) {
#ifdef _OPENMP
    if (!fftw_init_threads()) {
        return 1;
    }
    fftw_plan_with_nthreads(omt);
    return omt;
#endif
    return 1;
}

/**
 * @class FFTWPlanWrapper
 * @brief Template class to wrap FFTW plan details.
 *
 * @tparam T Data type of the buffer elements.
 * @tparam plan_t Type of the FFTW plan.
 */
template <class T, class plan_t> class FFTWPlanWrapper {
  public:
    int nFFTs;     /**< Number of FFTs */
    T* data;       /**< Pointer to the data buffer */
    T* scratch;    /**< Pointer to the scratch buffer */
    plan_t plan;   /**< FFTW plan */
    int direction; /**< FFT direction (FFTW_FORWARD or FFTW_BACKWARD) */
    int ng;        /**< Number of grid cells */
    bool valid;    /**< Plan validity flag */
};

/**
 * @class FFTWPlanManager
 * @brief Class to manage FFTW plans and perform FFT operations.
 */
class FFTWPlanManager : public FFTBackend_T {
  private:
    void* h_data;     /**< Pointer to the host data buffer */
    void* h_scratch;  /**< Pointer to the host scratch buffer */
    size_t last_size; /**< Size of the last allocated buffer */

    /**
     * @brief Allocate memory for host buffers.
     *
     * @param sz Size of the buffer to allocate.
     */
    void allocate_host(size_t sz);

  public:
    /**
     * @brief Array of cached FFTW plans for double-precision data.
     */
    FFTWPlanWrapper<fftw_complex, fftw_plan> double_plans[N_FFT_CACHE];
    /**
     * @brief Array of cached FFTW plans for single-precision data.
     */
    FFTWPlanWrapper<fftwf_complex, fftwf_plan> float_plans[N_FFT_CACHE];

    /**
     * @brief Query the FFTW backend.
     */
    void query();

    /**
     * @brief Constructor for FFTWPlanManager.
     */
    FFTWPlanManager();

    /**
     * @brief Destructor for FFTWPlanManager.
     */
    ~FFTWPlanManager();

    /**
     * @brief Find or create an FFTW plan for double-precision data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs.
     * @param direction FFT direction (FFTW_FORWARD or FFTW_BACKWARD).
     * @return fftw_plan Handle to the FFTW plan.
     */
    fftw_plan find_plan(fftw_complex* data, fftw_complex* scratch, int ng,
                        int nFFTs, int direction);

    /**
     * @brief Find or create an FFTW plan for single-precision data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs.
     * @param direction FFT direction (FFTW_FORWARD or FFTW_BACKWARD).
     * @return fftwf_plan Handle to the FFTW plan.
     */
    fftwf_plan find_plan(fftwf_complex* data, fftwf_complex* scratch, int ng,
                         int nFFTs, int direction);

    /**
     * @brief Perform forward FFT on host with double-precision FFTW data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs);

    /**
     * @brief Perform forward FFT on host with single-precision FFTW data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(fftwf_complex* data, fftwf_complex* scratch, int ng,
                 int nFFTs);

    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexDoubleHost* data, complexDoubleHost* scratch, int ng,
                 int nFFTs);

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexFloatHost* data, complexFloatHost* scratch, int ng,
                 int nFFTs);

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch,
                 int ng, int nFFTs);

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng,
                 int nFFTs);
#endif // SWFFT_GPU
    /**
     * @brief Perform backward FFT on host with double-precision FFTW data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs);

    /**
     * @brief Perform backward FFT on host with single-precision FFTW data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(fftwf_complex* data, fftwf_complex* scratch, int ng,
                  int nFFTs);

    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexDoubleHost* data, complexDoubleHost* scratch, int ng,
                  int nFFTs);

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexFloatHost* data, complexFloatHost* scratch, int ng,
                  int nFFTs);

#ifdef SWFFT_GPU
    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch,
                  int ng, int nFFTs);

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     * @param ng Number of grid cells.
     * @param nFFTs Number of FFTs to perform.
     */
    void backward(complexFloatDevice* data, complexFloatDevice* scratch, int ng,
                  int nFFTs);
#endif // SWFFT_GPU
};

/**
 * @typedef fftw
 * @brief Typedef for FFTWPlanManager.
 */
typedef FFTWPlanManager fftw;

/**
 * @brief Query the name of the fftw class.
 *
 * @return const char* The name of the fftw class.
 */
template <> inline const char* queryName<fftw>() { return "fftw"; }

#endif // SWFFT_FFTW
} // namespace SWFFT

#endif // _SWFFT_FFTW_BACKEND_HPP_