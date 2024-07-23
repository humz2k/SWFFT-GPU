/**
 * @file alltoall/dfft.hpp
 * @brief Header file for All-to-All distributed Fast Fourier Transform (DFFT)
 * operations in the SWFFT library.
 */

#ifndef _SWFFT_ALLTOALL_DFFT_HPP_
#define _SWFFT_ALLTOALL_DFFT_HPP_
#ifdef SWFFT_ALLTOALL

#include "alltoall/reorder.hpp"
#include "distribution.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "logging.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>

namespace SWFFT {
namespace A2A {

class alltoall_dist3d {
  private:
    bool m_ks_as_block;
    int m_local_grid_size[3];
    int m_local_coordinates_start[3];
    int m_nlocal;
    int m_world_size;
    int m_dims[3];
    int m_my_rank;

  public:
    alltoall_dist3d(bool ks_as_block, int local_grid_size[],
                    int local_coordinates_start[], int nlocal, int world_size,
                    int dims[], MPI_Comm comm)
        : m_ks_as_block(ks_as_block), m_local_grid_size{local_grid_size[0],
                                                        local_grid_size[1],
                                                        local_grid_size[2]},
          m_local_coordinates_start{local_coordinates_start[0],
                                    local_coordinates_start[1],
                                    local_coordinates_start[2]},
          m_nlocal(nlocal),
          m_world_size(world_size), m_dims{dims[0], dims[1], dims[2]} {
        MPI_Comm_rank(comm, &m_my_rank);
    }

#ifdef SWFFT_GPU
    __host__ __device__
#endif
        int3
        get_rs(int idx) {
        int3 local_idx;
        local_idx.x = idx / (m_local_grid_size[1] * m_local_grid_size[2]);
        local_idx.y = (idx - local_idx.x * (m_local_grid_size[1] *
                                            m_local_grid_size[2])) /
                      m_local_grid_size[2];
        local_idx.z = (idx - local_idx.x * (m_local_grid_size[1] *
                                            m_local_grid_size[2])) -
                      (local_idx.y * m_local_grid_size[2]);
        int3 global_idx = make_int3(m_local_coordinates_start[0] + local_idx.x,
                                    m_local_coordinates_start[1] + local_idx.y,
                                    m_local_coordinates_start[2] + local_idx.z);
        return global_idx;
    }

#ifdef SWFFT_GPU
    __host__ __device__
#endif
        int3
        get_ks(int idx) {
        if (m_ks_as_block) {
            int3 local_idx;
            local_idx.x = idx / (m_local_grid_size[1] * m_local_grid_size[2]);
            local_idx.y = (idx - local_idx.x * (m_local_grid_size[1] *
                                                m_local_grid_size[2])) /
                          m_local_grid_size[2];
            local_idx.z = (idx - local_idx.x * (m_local_grid_size[1] *
                                                m_local_grid_size[2])) -
                          (local_idx.y * m_local_grid_size[2]);
            int3 global_idx =
                make_int3(m_local_coordinates_start[0] + local_idx.x,
                          m_local_coordinates_start[1] + local_idx.y,
                          m_local_coordinates_start[2] + local_idx.z);
            return global_idx;
        }

        int i;

        // this is really really dumb please fix
        for (i = 0; i < m_nlocal; i++) {
            if ((A2A::calc_mini_pencil_idx(
                    i, (m_nlocal / m_world_size) / m_local_grid_size[1],
                    m_world_size, m_local_grid_size[1])) == idx)
                break;
        }

        int rank_of_origin = i / (m_nlocal / m_world_size);
        int rank_z = rank_of_origin / (m_dims[0] * m_dims[1]);
        int rank_x =
            (rank_of_origin - rank_z * m_dims[0] * m_dims[1]) / m_dims[1];
        int rank_y = (rank_of_origin - rank_z * m_dims[0] * m_dims[1]) -
                     rank_x * m_dims[1];

        int local_rank_idx = (i % (m_nlocal / m_world_size)) +
                             (m_nlocal / m_world_size) * m_my_rank;

        int lgridz = m_local_grid_size[2];
        int lgridy = m_local_grid_size[1];

        int _i = local_rank_idx / (lgridz * lgridy);
        int _k = (local_rank_idx - _i * (lgridz * lgridy)) / lgridy;
        int _j = (local_rank_idx - _i * (lgridz * lgridy)) - _k * lgridy;
        int dest_index = _i * lgridz * lgridy + _j * lgridz + _k;

        int3 local_idx;
        local_idx.x =
            dest_index / (m_local_grid_size[1] * m_local_grid_size[2]);
        local_idx.y = (dest_index - local_idx.x * (m_local_grid_size[1] *
                                                   m_local_grid_size[2])) /
                      m_local_grid_size[2];
        local_idx.z = (dest_index - local_idx.x * (m_local_grid_size[1] *
                                                   m_local_grid_size[2])) -
                      (local_idx.y * m_local_grid_size[2]);

        int3 global_idx;
        global_idx.x = rank_x * m_local_grid_size[0] + local_idx.x;
        global_idx.y = rank_y * m_local_grid_size[1] + local_idx.y;
        global_idx.z = rank_z * m_local_grid_size[2] + local_idx.z;

        return global_idx;
    }
};

/**
 * @class Dfft
 * @brief Class for managing All-to-All distributed Fast Fourier Transform
 * (DFFT) operations.
 *
 * @tparam MPI_T MPI wrapper type.
 * @tparam REORDER_T Reordering strategy type.
 * @tparam FFTBackend FFT backend type.
 */
template <class MPI_T, class REORDER_T, class FFTBackend> class Dfft {
  private:
    /**
     * @brief Perform FFT operation.
     *
     * @tparam T Data type of the buffer.
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     * @param direction Direction of the FFT (forward or backward).
     */
    template <class T> void fft(T* data, T* scratch, fftdirection direction);
    bool ks_as_block; /**< Flag indicating if k-space should be a block */

  public:
    int ng[3];      /**< Number of grid cells in each dimension */
    int nlocal;     /**< Number of local grid cells */
    int world_size; /**< Size of the MPI world */
    int world_rank; /**< Rank of the current process */
    int blockSize;  /**< Block size for operations */

    Distribution<MPI_T, REORDER_T>& dist; /**< Distribution strategy instance */
    FFTBackend FFTs;                      /**< FFT backend instance */

    alltoall_dist3d m_dist3d;

    /**
     * @brief Constructor for Dfft.
     *
     * @param dist_ Distribution strategy instance.
     * @param ks_as_block_ Flag indicating if k-space should be treated as a
     * block.
     */
    Dfft(Distribution<MPI_T, REORDER_T>& dist_, bool ks_as_block_);

    /**
     * @brief Destructor for Dfft.
     */
    ~Dfft();

    alltoall_dist3d dist3d();

    /**
     * @brief Get k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    int3 get_ks(int idx);

    /**
     * @brief Get real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    int3 get_rs(int idx);

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
#endif
    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexFloatHost* Buff1, complexFloatHost* Buff2);

#ifdef SWFFT_GPU
    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
#endif
    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexFloatHost* Buff1, complexFloatHost* Buff2);
};

} // namespace A2A
} // namespace SWFFT

#endif // SWFFT_ALLTOALL
#endif // _SWFFT_ALLTOALL_DFFT_HPP_