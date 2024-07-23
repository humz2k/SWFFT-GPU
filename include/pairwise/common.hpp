/**
 * @file pairwise/common.hpp
 * @brief Header file for common definitions and structures used in pairwise
 * communication in the SWFFT namespace.
 */

#ifndef _SWFFT_PAIRWISE_COMMON_HPP_
#define _SWFFT_PAIRWISE_COMMON_HPP_

#include "fftbackends/fftwrangler.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include <mpi.h>

namespace SWFFT {
namespace PAIR {

/**
 * @enum redist_t
 * @brief Enumeration of redistribution types for pairwise communication.
 */
enum redist_t {
    REDISTRIBUTE_1_TO_3, /**< Redistribution from 1D to 3D */
    REDISTRIBUTE_3_TO_1, /**< Redistribution from 3D to 1D */
    REDISTRIBUTE_2_TO_3, /**< Redistribution from 2D to 3D */
    REDISTRIBUTE_3_TO_2  /**< Redistribution from 3D to 2D */
};

/**
 * @struct process_topology_t
 * @brief Structure to define the process topology for pairwise communication.
 */
struct process_topology_t {
    MPI_Comm cart; /**< Cartesian communicator */
    int nproc[3];  /**< Number of processes in each dimension */
    int period[3]; /**< Periodicity of each dimension */
    int self[3];   /**< Coordinates of the current process */
    int n[3];      /**< Number of grid cells in each dimension */
};

} // namespace PAIR
} // namespace SWFFT

#endif // _SWFFT_PAIRWISE_COMMON_HPP_