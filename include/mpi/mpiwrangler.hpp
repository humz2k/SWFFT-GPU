/**
 * @file mpiwrangler.hpp
 * @brief Header file for MPI wrangling classes and functions in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_MPIWRANGLER_HPP_
#define _SWFFT_MPIWRANGLER_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"
#include "query.hpp"
#include <mpi.h>

#include "cpumpi.hpp"
#include "gpumpi.hpp"

namespace SWFFT {

/**
 * @typedef OPTMPI
 * @brief Typedef for selecting the appropriate MPI implementation (GPUMPI or
 * CPUMPI).
 *
 * This typedef selects GPUMPI if SWFFT_GPU is defined and SWFFT_NOCUDAMPI is
 * not defined. Otherwise, it selects CPUMPI.
 */
#if defined(SWFFT_GPU) && !defined(SWFFT_NOCUDAMPI)
typedef GPUMPI OPTMPI;
#else
typedef CPUMPI OPTMPI;
#endif

} // namespace SWFFT
#endif // _SWFFT_MPIWRANGLER_HPP_