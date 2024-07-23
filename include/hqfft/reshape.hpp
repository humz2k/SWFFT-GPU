/**
 * @file hqfft/reshape.hpp
 * @brief Header file for reshape operations in the SWFFT :: HQFFT namespace.
 */

#ifndef _SWFFT_HQFFT_RESHAPE_HPP_
#define _SWFFT_HQFFT_RESHAPE_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"

namespace SWFFT {
namespace HQFFT {

/**
 * @class Reshape
 * @brief Abstract base class for reshape operations.
 */
class Reshape {
  public:
    /**
     * @brief Constructor for Reshape.
     */
    Reshape() {}

    /**
     * @brief Virtual destructor for Reshape.
     */
    virtual ~Reshape() {}
#ifdef SWFFT_GPU
    virtual void reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize) = 0;

    virtual void unreshape(complexDoubleDevice* buff1,
                           complexDoubleDevice* buff2, int z_dim, int x_dim,
                           int y_dim, int nlocal, int blockSize) = 0;

    virtual void inverse_reshape(complexDoubleDevice* buff1,
                                 complexDoubleDevice* buff2, int n_recvs,
                                 int mini_pencil_size, int send_per_rank,
                                 int pencils_per_rank, int nlocal,
                                 int blockSize) = 0;

    virtual void inverse_unreshape(complexDoubleDevice* buff1,
                                   complexDoubleDevice* buff2, int z_dim,
                                   int x_dim, int y_dim, int nlocal,
                                   int blockSize) = 0;

    virtual void reshape_final(complexDoubleDevice* buff1,
                               complexDoubleDevice* buff2, int ny, int nz,
                               int local_grid_size[], int nlocal,
                               int blockSize) = 0;

    virtual void reshape(complexFloatDevice* buff1, complexFloatDevice* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize) = 0;
    virtual void unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2,
                           int z_dim, int x_dim, int y_dim, int nlocal,
                           int blockSize) = 0;
    virtual void inverse_reshape(complexFloatDevice* buff1,
                                 complexFloatDevice* buff2, int n_recvs,
                                 int mini_pencil_size, int send_per_rank,
                                 int pencils_per_rank, int nlocal,
                                 int blockSize) = 0;
    virtual void inverse_unreshape(complexFloatDevice* buff1,
                                   complexFloatDevice* buff2, int z_dim,
                                   int x_dim, int y_dim, int nlocal,
                                   int blockSize) = 0;
    virtual void reshape_final(complexFloatDevice* buff1,
                               complexFloatDevice* buff2, int ny, int nz,
                               int local_grid_size[], int nlocal,
                               int blockSize) = 0;
#endif
    virtual void reshape(complexDoubleHost* buff1, complexDoubleHost* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize) = 0;
    virtual void unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2,
                           int z_dim, int x_dim, int y_dim, int nlocal,
                           int blockSize) = 0;
    virtual void inverse_reshape(complexDoubleHost* buff1,
                                 complexDoubleHost* buff2, int n_recvs,
                                 int mini_pencil_size, int send_per_rank,
                                 int pencils_per_rank, int nlocal,
                                 int blockSize) = 0;
    virtual void inverse_unreshape(complexDoubleHost* buff1,
                                   complexDoubleHost* buff2, int z_dim,
                                   int x_dim, int y_dim, int nlocal,
                                   int blockSize) = 0;
    virtual void reshape_final(complexDoubleHost* buff1,
                               complexDoubleHost* buff2, int ny, int nz,
                               int local_grid_size[], int nlocal,
                               int blockSize) = 0;

    virtual void reshape(complexFloatHost* buff1, complexFloatHost* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize) = 0;
    virtual void unreshape(complexFloatHost* buff1, complexFloatHost* buff2,
                           int z_dim, int x_dim, int y_dim, int nlocal,
                           int blockSize) = 0;
    virtual void inverse_reshape(complexFloatHost* buff1,
                                 complexFloatHost* buff2, int n_recvs,
                                 int mini_pencil_size, int send_per_rank,
                                 int pencils_per_rank, int nlocal,
                                 int blockSize) = 0;
    virtual void inverse_unreshape(complexFloatHost* buff1,
                                   complexFloatHost* buff2, int z_dim,
                                   int x_dim, int y_dim, int nlocal,
                                   int blockSize) = 0;
    virtual void reshape_final(complexFloatHost* buff1, complexFloatHost* buff2,
                               int ny, int nz, int local_grid_size[],
                               int nlocal, int blockSize) = 0;
};

} // namespace HQFFT
} // namespace SWFFT
#endif // _SWFFT_HQFFT_RESHAPE_HPP_