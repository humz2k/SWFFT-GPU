/**
 * @file gpumpi.hpp
 * @brief Header file for GPUMPI wrangling classes and functions in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_GPUMPI_HPP_
#define _SWFFT_GPUMPI_HPP_
#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI

#include "complex-type.hpp"
#include "gpu.hpp"
#include "query.hpp"
#include "xpumpi.hpp"
#include <mpi.h>

namespace SWFFT {
/**
 * @class GPUIsend
 * @brief Class for non-blocking MPI send operations on the GPU.
 *
 * @tparam T Data type of the buffer being sent.
 */
template <class T> class GPUIsend : public XPUIsend<T> {
  public:
    using XPUIsend<T>::XPUIsend;
};

/**
 * @class GPUIrecv
 * @brief Class for non-blocking MPI receive operations on the GPU.
 *
 * @tparam T Data type of the buffer being received.
 */
template <class T> class GPUIrecv : public XPUIrecv<T> {
  public:
    using XPUIrecv<T>::XPUIrecv;
};

/**
 * @class GPUMPI
 * @brief Class for GPU-based MPI operations.
 */
class GPUMPI : public XPUMPI<GPUIsend, GPUIrecv> {
  public:
    using XPUMPI::XPUMPI;

    /**
     * @brief Query the MPI implementation being used.
     */
    void query() { printf("Using GPUMPI\n"); };
};

template <> inline const char* queryName<GPUMPI>() { return "GPUMPI"; }

} // namespace SWFFT

#endif // ~SWFFT_NOCUDAMPI
#endif // SWFFT_GPU
#endif // _SWFFT_GPUMPI_HPP_