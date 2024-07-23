/**
 * @file pairwise/dfft.hpp
 * @brief Header file for distributed FFT classes and functions in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_PAIRWISE_DFFT_HPP_
#define _SWFFT_PAIRWISE_DFFT_HPP_

#include "common.hpp"
#include "distribution.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include <mpi.h>

namespace SWFFT {
namespace PAIR {

class pairwiseDist3d {
  private:
    process_topology_t m_process_topology_2_z; /**< 2D dist (z) */
    process_topology_t m_process_topology_3;   /**< 3D dist */
  public:
    pairwiseDist3d(process_topology_t process_topology_2_z,
                    process_topology_t process_topology_3)
        : m_process_topology_2_z(process_topology_2_z),
          m_process_topology_3(process_topology_3) {}

#ifdef SWFFT_GPU
    __host__ __device__
#endif
        int3
        get_ks(int idx) {
        int3 local_ng_k =
            make_int3(m_process_topology_2_z.n[0], m_process_topology_2_z.n[1],
                      m_process_topology_2_z.n[2]);
        int3 pos_k = make_int3(m_process_topology_2_z.self[0],
                               m_process_topology_2_z.self[1],
                               m_process_topology_2_z.self[2]);
        int3 my_pos;
        my_pos.x = idx / (local_ng_k.y * local_ng_k.z);
        my_pos.y =
            (idx - (my_pos.x * local_ng_k.y * local_ng_k.z)) / local_ng_k.z;
        my_pos.z = (idx - (my_pos.x * local_ng_k.y * local_ng_k.z)) -
                   my_pos.y * local_ng_k.z;
        my_pos.x += pos_k.x * local_ng_k.x;
        my_pos.y += pos_k.y * local_ng_k.y;
        my_pos.z += pos_k.z * local_ng_k.z;
        return my_pos;
    }

#ifdef SWFFT_GPU
    __host__ __device__
#endif
        int3
        get_rs(int idx) {
        int3 local_ng_k =
            make_int3(m_process_topology_3.n[0], m_process_topology_3.n[1],
                      m_process_topology_3.n[2]);
        int3 pos_r = make_int3(m_process_topology_3.self[0],
                               m_process_topology_3.self[1],
                               m_process_topology_3.self[2]);
        int3 my_pos;
        my_pos.x = idx / (local_ng_k.y * local_ng_k.z);
        my_pos.y =
            (idx - (my_pos.x * local_ng_k.y * local_ng_k.z)) / local_ng_k.z;
        my_pos.z = (idx - (my_pos.x * local_ng_k.y * local_ng_k.z)) -
                   my_pos.y * local_ng_k.z;
        my_pos.x += pos_r.x * local_ng_k.x;
        my_pos.y += pos_r.y * local_ng_k.y;
        my_pos.z += pos_r.z * local_ng_k.z;
        return my_pos;
    }
};

/**
 * @class Dfft
 * @brief Class to manage distributed FFT operations using MPI and FFT backends.
 *
 * @tparam MPI_T MPI implementation type (e.g., CPUMPI, GPUMPI).
 * @tparam FFTBackend FFT backend type (e.g., FFTW, cuFFT).
 */
template <class MPI_T, class FFTBackend> class Dfft {
  private:
    MPI_Comm m_comm;   /**< MPI communicator */
    FFTBackend m_FFTs; /**< FFT backend */
    int m_n[3];        /**< Dimensions of the data grid */
    distribution_t<complexDoubleHost, MPI_T> m_double_dist; /**< Dist (double) */
    distribution_t<complexFloatHost, MPI_T> m_float_dist;   /**< Dist (single) */
    pairwiseDist3d m_dist3d;

    /**
     * @brief Template method to perform forward FFT.
     *
     * @tparam T Data type of the buffer elements.
     * @param data Pointer to the data buffer.
     */
    template <class T> void _forward(T* data);

    /**
     * @brief Template method to perform backward FFT.
     *
     * @tparam T Data type of the buffer elements.
     * @param data Pointer to the data buffer.
     */
    template <class T> void _backward(T* data);

  public:
    /**
     * @brief Constructor for Dfft.
     *
     * @param comm_ MPI communicator.
     * @param nx Number of grid cells in the x dimension.
     * @param ny Number of grid cells in the y dimension.
     * @param nz Number of grid cells in the z dimension.
     */
    Dfft(MPI_Comm comm_, int nx, int ny, int nz);

    /**
     * @brief Destructor for Dfft.
     */
    ~Dfft();

    /**
     * @brief Get the buffer size required for FFT operations.
     *
     * @return int Buffer size.
     */
    int buff_sz();

    /**
     * @brief Get the coordinates of the current process.
     *
     * @return int3 Coordinates of the current process.
     */
    int3 coords();

    pairwiseDist3d dist3d();

    /**
     * @brief Get the k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    int3 get_ks(int idx);

    /**
     * @brief Get the real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    int3 get_rs(int idx);

    /**
     * @brief Get the number of processes in 3D distribution for a given
     * direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Number of processes.
     */
    int get_nproc_3d(int direction);

    /**
     * @brief Get the local number of grid cells in 3D for a given direction.
     *
     * @param direction Direction index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int get_local_ng_3d(int direction);

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void forward(complexDoubleDevice* data);

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void forward(complexFloatDevice* data);

    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void backward(complexDoubleDevice* data);

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void backward(complexFloatDevice* data);

    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch);

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexFloatDevice* data, complexFloatDevice* scratch);

    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch);

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexFloatDevice* data, complexFloatDevice* scratch);
#endif

    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexDoubleHost* data, complexDoubleHost* scratch);

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexFloatHost* data, complexFloatHost* scratch);

    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexDoubleHost* data, complexDoubleHost* scratch);

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexFloatHost* data, complexFloatHost* scratch);

    /**
     * @brief Perform forward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void forward(complexDoubleHost* data);

    /**
     * @brief Perform forward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void forward(complexFloatHost* data);

    /**
     * @brief Perform backward FFT on host with double-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void backward(complexDoubleHost* data);

    /**
     * @brief Perform backward FFT on host with single-precision complex data.
     *
     * @param data Pointer to the data buffer.
     */
    void backward(complexFloatHost* data);
};

} // namespace PAIR
} // namespace SWFFT

#endif // _SWFFT_PAIRWISE_DFFT_HPP_