/**
 * @file hqfft/gpureshape.hpp
 */
#ifndef _SWFFT_HQFFT_GPURESHAPE_HPP_
#define _SWFFT_HQFFT_GPURESHAPE_HPP_
#ifdef SWFFT_GPU

#include "complex-type.hpp"
#include "gpu.hpp"
#include "reshape.hpp"

namespace SWFFT {
namespace HQFFT {

class GPUReshape : public Reshape {
  private:
    template <class T>
    void _reshape(T* buff1, T* buff2, int n_recvs, int mini_pencil_size,
                  int send_per_rank, int pencils_per_rank, int nlocal,
                  int blockSize);

    template <class T>
    void _inverse_reshape(T* buff1, T* buff2, int n_recvs, int mini_pencil_size,
                          int send_per_rank, int pencils_per_rank, int nlocal,
                          int blockSize);

    template <class T>
    void _unreshape(T* buff1, T* buff2, int z_dim, int x_dim, int y_dim,
                    int nlocal, int blockSize);

    template <class T>
    void _inverse_unreshape(T* buff1, T* buff2, int z_dim, int x_dim, int y_dim,
                            int nlocal, int blockSize);

    template <class T>
    void _reshape_final(T* buff1, T* buff2, int ny, int nz,
                        int local_grid_size[], int nlocal, int blockSize);

  public:
    GPUReshape();
    ~GPUReshape();

    void reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                 int n_recvs, int mini_pencil_size, int send_per_rank,
                 int pencils_per_rank, int nlocal, int blockSize);
    void unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                   int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
    void inverse_reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize);
    void inverse_unreshape(complexDoubleDevice* buff1,
                           complexDoubleDevice* buff2, int z_dim, int x_dim,
                           int y_dim, int nlocal, int blockSize);
    void reshape_final(complexDoubleDevice* buff1, complexDoubleDevice* buff2,
                       int ny, int nz, int local_grid_size[], int nlocal,
                       int blockSize);

    void reshape(complexFloatDevice* buff1, complexFloatDevice* buff2,
                 int n_recvs, int mini_pencil_size, int send_per_rank,
                 int pencils_per_rank, int nlocal, int blockSize);
    void unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2,
                   int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
    void inverse_reshape(complexFloatDevice* buff1, complexFloatDevice* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize);
    void inverse_unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2,
                           int z_dim, int x_dim, int y_dim, int nlocal,
                           int blockSize);
    void reshape_final(complexFloatDevice* buff1, complexFloatDevice* buff2,
                       int ny, int nz, int local_grid_size[], int nlocal,
                       int blockSize);

    void reshape(complexDoubleHost* buff1, complexDoubleHost* buff2,
                 int n_recvs, int mini_pencil_size, int send_per_rank,
                 int pencils_per_rank, int nlocal, int blockSize);
    void unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2,
                   int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
    void inverse_reshape(complexDoubleHost* buff1, complexDoubleHost* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize);
    void inverse_unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2,
                           int z_dim, int x_dim, int y_dim, int nlocal,
                           int blockSize);
    void reshape_final(complexDoubleHost* buff1, complexDoubleHost* buff2,
                       int ny, int nz, int local_grid_size[], int nlocal,
                       int blockSize);

    void reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs,
                 int mini_pencil_size, int send_per_rank, int pencils_per_rank,
                 int nlocal, int blockSize);
    void unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim,
                   int x_dim, int y_dim, int nlocal, int blockSize);
    void inverse_reshape(complexFloatHost* buff1, complexFloatHost* buff2,
                         int n_recvs, int mini_pencil_size, int send_per_rank,
                         int pencils_per_rank, int nlocal, int blockSize);
    void inverse_unreshape(complexFloatHost* buff1, complexFloatHost* buff2,
                           int z_dim, int x_dim, int y_dim, int nlocal,
                           int blockSize);
    void reshape_final(complexFloatHost* buff1, complexFloatHost* buff2, int ny,
                       int nz, int local_grid_size[], int nlocal,
                       int blockSize);
};

} // namespace HQFFT
} // namespace SWFFT
#endif // SWFFT_GPU
#endif // _SWFFT_HQFFT_GPURESHAPE_HPP_