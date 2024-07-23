/**
 * @file pairwise/distribution.hpp
 * @brief Header file for pairwise distribution classes and functions in the
 * SWFFT namespace.
 */

#ifndef _SWFFT_PAIRWISE_DISTRIBUTION_HPP_
#define _SWFFT_PAIRWISE_DISTRIBUTION_HPP_

#include "fftbackends/fftwrangler.hpp"
#include "mpi/mpiwrangler.hpp"
#include "common.hpp"
#include "query.hpp"
#include <mpi.h>

namespace SWFFT {
namespace PAIR {

/**
 * @class distribution_t
 * @brief Class to manage pairwise distribution.
 *
 * @tparam T Data type of the buffer elements.
 * @tparam MPI_T MPI implementation type (e.g., CPUMPI, GPUMPI).
 */
template <class T, class MPI_T> class distribution_t {
  public:
    bool debug; /**< Debug flag */
    int n[3];   /**< Dimensions of the data grid */

    MPI_T mpi; /**< MPI implementation */

    process_topology_t process_topology_1;   /**< 1D dist */
    process_topology_t process_topology_2_z; /**< 2D dist (z) */
    process_topology_t process_topology_2_y; /**< 2D dist (y) */
    process_topology_t process_topology_2_x; /**< 2D dist (x) */
    process_topology_t process_topology_3;   /**< 3D dist */

    T* d2_chunk;     /**< Pointer to 2D chunk of data */
    T* d3_chunk;     /**< Pointer to 3D chunk of data */
    MPI_Comm parent; /**< Parent MPI communicator */

    /**
     * @brief Constructor for distribution_t.
     *
     * @param comm Parent MPI communicator.
     * @param nx Number of grid cells in the x dimension.
     * @param ny Number of grid cells in the y dimension.
     * @param nz Number of grid cells in the z dimension.
     * @param debug_ Debug flag.
     */
    distribution_t(MPI_Comm comm, int nx, int ny, int nz, bool debug_);

    /**
     * @brief Destructor for distribution_t.
     */
    ~distribution_t();

    /**
     * @brief Assert that the process topology is commensurate.
     */
    void assert_commensurate();

    /**
     * @brief Redistribute data between different process topologies.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     * @param r Redistribution type.
     */
    void redistribute(const T* a, T* b, redist_t r);

    /**
     * @brief Redistribute data between 2D and 3D process topologies.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     * @param r Redistribution type.
     * @param z_dim Dimension to use for the redistribution.
     */
    void redistribute_2_and_3(const T* a, T* b, redist_t r, int z_dim);

    /**
     * @brief Redistribute data in slab format.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     * @param r Redistribution type.
     */
    void redistribute_slab(const T* a, T* b, redist_t r);

    /**
     * @brief Redistribute data from 1D to 3D process topology.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     */
    void dist_1_to_3(const T* a, T* b);

    /**
     * @brief Redistribute data from 3D to 1D process topology.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     */
    void dist_3_to_1(const T* a, T* b);

    /**
     * @brief Redistribute data from 2D to 3D process topology.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     * @param dim_z Dimension to use for the redistribution.
     */
    void dist_2_to_3(const T* a, T* b, int dim_z);

    /**
     * @brief Redistribute data from 3D to 2D process topology.
     *
     * @param a Pointer to the source data buffer.
     * @param b Pointer to the destination data buffer.
     * @param dim_z Dimension to use for the redistribution.
     */
    void dist_3_to_2(const T* a, T* b, int dim_z);

    /**
     * @brief Get the number of processes in 1D distribution for a given
     * direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Number of processes.
     */
    int get_nproc_1d(int direction);

    /**
     * @brief Get the number of processes in 2D distribution (x dimension) for a
     * given direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Number of processes.
     */
    int get_nproc_2d_x(int direction);

    /**
     * @brief Get the number of processes in 2D distribution (y dimension) for a
     * given direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Number of processes.
     */
    int get_nproc_2d_y(int direction);

    /**
     * @brief Get the number of processes in 2D distribution (z dimension) for a
     * given direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Number of processes.
     */
    int get_nproc_2d_z(int direction);

    /**
     * @brief Get the number of processes in 3D distribution for a given
     * direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Number of processes.
     */
    int get_nproc_3d(int direction);

    /**
     * @brief Get the process rank in 1D distribution for a given direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Process rank.
     */
    int get_self_1d(int direction);

    /**
     * @brief Get the process rank in 2D distribution (x dimension) for a given
     * direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Process rank.
     */
    int get_self_2d_x(int direction);

    /**
     * @brief Get the process rank in 2D distribution (y dimension) for a given
     * direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Process rank.
     */
    int get_self_2d_y(int direction);

    /**
     * @brief Get the process rank in 2D distribution (z dimension) for a given
     * direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Process rank.
     */
    int get_self_2d_z(int direction);

    /**
     * @brief Get the process rank in 3D distribution for a given direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Process rank.
     */
    int get_self_3d(int direction);

    void coord_x_pencils(int myrank, int coord[]);

    void rank_x_pencils(int* myrank, int coord[]);

    int rank_x_pencils(int coord[]);

    void coord_y_pencils(int myrank, int coord[]);

    void rank_y_pencils(int* myrank, int coord[]);

    int rank_y_pencils(int coord[]);

    void coord_z_pencils(int myrank, int coord[]);

    void rank_z_pencils(int* myrank, int coord[]);

    int rank_z_pencils(int coord[]);

    void coord_cube(int myrank, int coord[]);

    void rank_cube(int* myrank, int coord[]);

    int rank_cube(int coord[]);

    /**
     * @brief Get the local number of grid cells in 1D for a given direction.
     *
     * @param i Direction index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng_1d(int i);

    /**
     * @brief Get the local number of grid cells in 2D (x dimension) for a given
     * direction.
     *
     * @param i Direction index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng_2d_x(int i);

    /**
     * @brief Get the local number of grid cells in 2D (y dimension) for a given
     * direction.
     *
     * @param i Direction index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng_2d_y(int i);

    /**
     * @brief Get the local number of grid cells in 2D (z dimension) for a given
     * direction.
     *
     * @param i Direction index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng_2d_z(int i);

    /**
     * @brief Get the local number of grid cells in 3D for a given direction.
     *
     * @param i Direction index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng_3d(int i);
};

} // namespace PAIR
} // namespace SWFFT

#endif // _SWFFT_PAIRWISE_DISTRIBUTION_HPP_