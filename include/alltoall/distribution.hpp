/**
 * @file alltoall/distribution.hpp
 * @brief Header file for All-to-All distribution operations in the SWFFT
 * library.
 */

#ifndef _SWFFT_ALLTOALL_DISTRIBUTION_HPP_
#define _SWFFT_ALLTOALL_DISTRIBUTION_HPP_
#ifdef SWFFT_ALLTOALL

#include "alltoall/reorder.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "logging.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>

namespace SWFFT {
namespace A2A {
/**
 * @class Distribution
 * @brief Class for managing the distribution of data in All-to-All
 * communication patterns.
 *
 * @tparam MPI_T MPI wrapper type.
 * @tparam REORDER_T Reordering strategy type.
 */
template <class MPI_T, class REORDER_T> class Distribution {
  public:
    int ndims;              /**< Number of dimensions */
    int ng[3];              /**< Number of grid cells in each dimension */
    int nlocal;             /**< Number of local grid cells */
    int world_size;         /**< Size of the MPI world */
    int world_rank;         /**< Rank of the current process */
    int local_grid_size[3]; /**< Size of the local grid */
    int dims[3];            /**< Dimensions of the process grid */
    int coords[3];          /**< Coordinates of the current process */
    int local_coordinates_start[3]; /**< Starting coordinates of the local grid
                                     */
    bool ks_as_block;     /**< Flag indicating if k-space should be treated as a
                             block */
    MPI_Comm comm;        /**< MPI communicator */
    MPI_Comm fftcomms[3]; /**< MPI communicators for FFT operations */

    MPI_T mpi;            /**< MPI wrapper instance */
    REORDER_T reordering; /**< Reordering strategy instance */

    bool use_alltoall[3]; /**< Flags for using All-to-All communication in each
                             dimension */

    int blockSize; /**< Block size for operations */

    /**
     * @brief Constructor for Distribution.
     *
     * @param comm_ MPI communicator.
     * @param ngx Number of grid cells in the x dimension.
     * @param ngy Number of grid cells in the y dimension.
     * @param ngz Number of grid cells in the z dimension.
     * @param blockSize_ Block size for operations.
     * @param ks_as_block_ Flag indicating if k-space should be a block.
     */
    Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_,
                 bool ks_as_block_);

    /**
     * @brief Destructor for Distribution.
     */
    ~Distribution();

    /**
     * @brief Assert the correctness of the distribution.
     *
     * @return int Status of the assertion.
     */
    int assert_distribution();

    /**
     * @brief Get the MPI communicator for the first shuffle.
     *
     * @return MPI_Comm MPI communicator.
     */
    MPI_Comm shuffle_comm_1();

    /**
     * @brief Get the MPI communicator for the second shuffle.
     *
     * @return MPI_Comm MPI communicator.
     */
    MPI_Comm shuffle_comm_2();

    /**
     * @brief Get the MPI communicator for a specific shuffle.
     *
     * @param n Shuffle index.
     * @return MPI_Comm MPI communicator.
     */
    MPI_Comm shuffle_comm(int n);

    /**
     * @brief Get pencils from the buffer.
     *
     * @tparam T Data type of the buffer.
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    template <class T> inline void getPencils_(T* Buff1, T* Buff2, int dim);

#ifdef SWFFT_GPU
    /**
     * @brief Copy data between device buffers.
     *
     * @param Buff1 Destination buffer.
     * @param Buff2 Source buffer.
     */
    void copy(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);

    /**
     * @brief Copy data between device buffers.
     *
     * @param Buff1 Destination buffer.
     * @param Buff2 Source buffer.
     */
    void copy(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
#endif
    /**
     * @brief Copy data between host buffers.
     *
     * @param Buff1 Destination buffer.
     * @param Buff2 Source buffer.
     */
    void copy(complexDoubleHost* __restrict Buff1,
              const complexDoubleHost* __restrict Buff2);

    /**
     * @brief Copy data between host buffers.
     *
     * @param Buff1 Destination buffer.
     * @param Buff2 Source buffer.
     */
    void copy(complexFloatHost* __restrict Buff1,
              const complexFloatHost* __restrict Buff2);

#ifdef SWFFT_GPU
    /**
     * @brief Get pencils from device buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void getPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                    int dim);

    /**
     * @brief Get pencils from device buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void getPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                    int dim);
#endif
    /**
     * @brief Get pencils from host buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void getPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                    int dim);

    /**
     * @brief Get pencils from host buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void getPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

    /**
     * @brief Return pencils to the buffer.
     *
     * @tparam T Data type of the buffer.
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    template <class T> inline void returnPencils_(T* Buff1, T* Buff2, int dim);

#ifdef SWFFT_GPU
    /**
     * @brief Return pencils to device buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void returnPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                       int dim);

    /**
     * @brief Return pencils to device buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void returnPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                       int dim);
#endif
    /**
     * @brief Return pencils to host buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void returnPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                       int dim);

    /**
     * @brief Return pencils to host buffers.
     *
     * @param Buff1 Input buffer.
     * @param Buff2 Output buffer.
     * @param dim Dimension index.
     */
    void returnPencils(complexFloatHost* Buff1, complexFloatHost* Buff2,
                       int dim);

#ifdef SWFFT_GPU
    /**
     * @brief Shuffle indices in device buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                         int n);

    /**
     * @brief Shuffle indices in device buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                         int n);
#endif
    /**
     * @brief Shuffle indices in host buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                         int n);

    /**
     * @brief Shuffle indices in host buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2,
                         int n);

#ifdef SWFFT_GPU
    /**
     * @brief Reorder device buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder device buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n,
                 int direction);
#endif
    /**
     * @brief Reorder host buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder host buffers.
     *
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n,
                 int direction);
};

} // namespace A2A
} // namespace SWFFT
#endif // SWFFT_ALLTOALL
#endif // _SWFFT_ALLTOALL_DISTRIBUTION_HPP_