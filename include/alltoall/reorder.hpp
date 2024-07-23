/**
 * @file alltoall/reorder.hpp
 * @brief Header file for All-to-All reordering operations in the SWFFT library.
 */

#ifndef _SWFFT_ALLTOALL_REORDER_HPP_
#define _SWFFT_ALLTOALL_REORDER_HPP_
#ifdef SWFFT_ALLTOALL

#include <stdio.h>
#include <stdlib.h>

#include "complex-type.hpp"
#include "gpu.hpp"

namespace SWFFT {

namespace A2A {
/**
 * @brief Calculate mini pencil index.
 * @param i Index.
 * @param mini_pencils_per_rank Number of mini pencils per rank.
 * @param world_size Size of the MPI world.
 * @param mini_pencil_size Size of the mini pencil.
 * @return int Mini pencil index.
 */
#ifdef SWFFT_GPU
static __forceinline__ __host__ __device__
#endif
    int
    calc_mini_pencil_idx(int i, int mini_pencils_per_rank, int world_size,
                         int mini_pencil_size) {
    int global_mini_pencil_id = i / mini_pencil_size;
    int rank = global_mini_pencil_id / mini_pencils_per_rank;
    int local_mini_pencil_id = global_mini_pencil_id % mini_pencils_per_rank;
    int global_mini_pencil_offset = world_size * mini_pencil_size;
    int my_mini_pencil_offset =
        local_mini_pencil_id * global_mini_pencil_offset;
    int my_pencil_start = rank * mini_pencil_size + my_mini_pencil_offset;
    int sub_mini_pencil_idx = i % mini_pencil_size;
    int new_idx = my_pencil_start + sub_mini_pencil_idx;
    return new_idx;
}

#ifdef SWFFT_GPU
/**
 * @class GPUReorder
 * @brief Class for GPU-based reordering operations.
 */
class GPUReorder {
  private:
    int3 m_ng;                /**< Number of grid cells in each dimension */
    int3 m_dims;              /**< Dimensions of the process grid */
    int3 m_coords;            /**< Coordinates of the current process */
    int3 m_local_grid;        /**< Local grid dimensions */
    int m_local_grid_size[3]; /**< Size of the local grid */
    int m_nlocal;             /**< Number of local grid cells */
    int m_world_size;         /**< Size of the MPI world */
    int m_blockSize;          /**< Block size for GPU operations */

    /**
     * @brief GPU-based shuffle indices operation.
     * @tparam T Data type of the buffer.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle (0: z->x, 1: x->y, 2: y->z, 3: x->z).
     */
    template <class T>
    inline void gpu_shuffle_indices(T* Buff1, T* Buff2, int n);

    /**
     * @brief GPU-based reorder operation.
     * @tparam T Data type of the buffer.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    template <class T>
    inline void gpu_reorder(T* Buff1, T* Buff2, int n, int direction);

  public:
    /**
     * @brief Default constructor for GPUReorder.
     */
    GPUReorder();

    /**
     * @brief Parameterized constructor for GPUReorder.
     * @param ng_ Number of grid cells in each dimension.
     * @param dims_ Dimensions of the process grid.
     * @param coords_ Coordinates of the current process.
     * @param blockSize_ Block size for GPU operations.
     */
    GPUReorder(int3 ng, int3 dims, int3 coords, int blockSize);

    /**
     * @brief Destructor for GPUReorder.
     */
    ~GPUReorder();

    /**
     * @brief Shuffle indices operation for complex double precision device
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                         int n);

    /**
     * @brief Shuffle indices operation for complex single precision device
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                         int n);

    /**
     * @brief Shuffle indices operation for complex double precision host
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                         int n);

    /**
     * @brief Shuffle indices operation for complex single precision host
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2,
                         int n);

    /**
     * @brief Reorder operation for complex double precision device buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder operation for complex single precision device buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder operation for complex double precision host buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder operation for complex single precision host buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n,
                 int direction);
};
#endif // SWFFT_GPU

/**
 * @class CPUReorder
 * @brief Class for CPU-based reordering operations.
 */
class CPUReorder {
  private:
    int3 m_ng;                /**< Number of grid cells in each dimension */
    int3 m_dims;              /**< Dimensions of the process grid */
    int3 m_coords;            /**< Coordinates of the current process */
    int3 m_local_grid;        /**< Local grid dimensions */
    int m_local_grid_size[3]; /**< Size of the local grid */
    int m_nlocal;             /**< Number of local grid cells */
    int m_world_size;         /**< Size of the MPI world */
    int m_blockSize;          /**< Block size for operations */

    /**
     * @brief CPU-based shuffle indices operation.
     * @tparam T Data type of the buffer.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    template <class T> void cpu_shuffle_indices(T* Buff1, T* Buff2, int n);

    /**
     * @brief CPU-based reorder operation.
     * @tparam T Data type of the buffer.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations (0, 1, or 2)
     * @param direction Reorder direction.
     */
    template <class T>
    void cpu_reorder(T* Buff1, T* Buff2, int n, int direction);

  public:
    /**
     * @brief Default constructor for CPUReorder.
     */
    CPUReorder();

    /**
     * @brief Parameterized constructor for CPUReorder.
     * @param ng_ Number of grid cells in each dimension.
     * @param dims_ Dimensions of the process grid.
     * @param coords_ Coordinates of the current process.
     * @param blockSize_ Block size for operations.
     */
    CPUReorder(int3 ng_, int3 dims_, int3 coords_, int blockSize_);

    /**
     * @brief Destructor for CPUReorder.
     */
    ~CPUReorder();

#ifdef SWFFT_GPU
    /**
     * @brief Shuffle indices operation for complex double precision device
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                         int n);

    /**
     * @brief Shuffle indices operation for complex single precision device
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                         int n);
#endif
    /**
     * @brief Shuffle indices operation for complex double precision host
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                         int n);

    /**
     * @brief Shuffle indices operation for complex single precision host
     * buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Indices to shuffle.
     */
    void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2,
                         int n);

#ifdef SWFFT_GPU
    /**
     * @brief Reorder operation for complex double precision device buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations
     * @param direction Reorder direction.
     */
    void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder operation for complex single precision device buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations
     * @param direction Reorder direction.
     */
    void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n,
                 int direction);
#endif
    /**
     * @brief Reorder operation for complex double precision host buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations
     * @param direction Reorder direction.
     */
    void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n,
                 int direction);

    /**
     * @brief Reorder operation for complex single precision host buffers.
     * @param Buff1 Output buffer.
     * @param Buff2 Input buffer.
     * @param n Reorder operations
     * @param direction Reorder direction.
     */
    void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n,
                 int direction);
};
} // namespace A2A

} // namespace SWFFT

#endif // SWFFT_ALLTOALL
#endif // _SWFFT_ALLTOALL_REORDER_HPP_