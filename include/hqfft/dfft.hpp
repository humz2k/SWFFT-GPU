/**
 * @file hqfft/dfft.hpp
 */

#ifndef _SWFFT_HQFFT_DFFT_HPP_
#define _SWFFT_HQFFT_DFFT_HPP_
#ifdef SWFFT_HQFFT

#include "collectivecomm.hpp"
#include "common/copy_buffers.hpp"
#include "distribution.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "hqfft_reorder.hpp"
#include "mpi/mpi_isend_irecv.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>
#include <stdio.h>

namespace SWFFT {
namespace HQFFT {

class hqfftDist3d {
  private:
    bool m_ks_as_block;
    int m_local_grid_size[3];
    int m_local_coords_start[3];
    int m_nlocal;
    int m_ng[3];
    int m_dims[3];
    int m_coords[3];

  public:
    hqfftDist3d(bool ks_as_block, int local_grid_size[],
                int local_coords_start[], int nlocal, int ng[], int dims[],
                int coords[])
        : m_ks_as_block(ks_as_block), m_local_grid_size{local_grid_size[0],
                                                        local_grid_size[1],
                                                        local_grid_size[2]},
          m_local_coords_start{local_coords_start[0], local_coords_start[1],
                               local_coords_start[2]},
          m_nlocal(nlocal), m_ng{ng[0], ng[1], ng[2]},
          m_dims{dims[0], dims[1], dims[2]}, m_coords{coords[0], coords[1],
                                                      coords[2]} {}

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
        int3 global_idx = make_int3(m_local_coords_start[0] + local_idx.x,
                                    m_local_coords_start[1] + local_idx.y,
                                    m_local_coords_start[2] + local_idx.z);
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
            int3 global_idx = make_int3(m_local_coords_start[0] + local_idx.x,
                                        m_local_coords_start[1] + local_idx.y,
                                        m_local_coords_start[2] + local_idx.z);
            return global_idx;
        } else {

            int z_dim = m_ng[0];
            int x_dim = m_local_grid_size[1] / m_dims[0];
            int y_dim = (m_nlocal / z_dim) / x_dim;

            int _x = idx / (y_dim * z_dim);
            int _y = (idx - (_x * y_dim * z_dim)) / z_dim;
            int _z = (idx - (_x * y_dim * z_dim)) - _y * z_dim;
            int new_idx = _z * x_dim * y_dim + _x * y_dim + _y;

            int dest_x_start = 0;

            int y = ((m_coords[0] * m_dims[2] + m_coords[2]) *
                     (m_ng[1] / (m_dims[0] * m_dims[2]))) /
                    m_local_grid_size[1];

            int y_send =
                m_local_grid_size[1] / (m_ng[1] / (m_dims[0] * m_dims[2]));

            int z =
                (m_coords[1] * (m_ng[2] / m_dims[1])) / m_local_grid_size[2];

            int n_recvs = m_dims[0];

            int count = new_idx / (m_nlocal / n_recvs);

            int x = dest_x_start + count;

            int3 global_idx =
                make_int3(x * m_local_grid_size[0], y * m_local_grid_size[1],
                          z * m_local_grid_size[2]);

            int ysrc = ((m_coords[0] * m_dims[2] + m_coords[2]) *
                        (m_ng[1] / (m_dims[0] * m_dims[2])));
            int zsrc = (m_coords[1] * (m_ng[2] / m_dims[1]));

            int zoff = zsrc - m_local_grid_size[2] * z;
            int yoff = (ysrc - m_local_grid_size[1] * y) /
                       (m_ng[1] / (m_dims[0] * m_dims[2]));
            int tmp1 = zoff / (m_ng[2] / m_dims[1]);
            int new_count = tmp1 * y_send + yoff;

            int i = (new_idx % (m_nlocal / n_recvs)) +
                    new_count * (m_nlocal / n_recvs);

            int ny = y_send;
            int nz = n_recvs / y_send;

            {
                int3 local_dims =
                    make_int3(m_local_grid_size[0], m_local_grid_size[1] / ny,
                              m_local_grid_size[2] / nz); // per rank dims

                int n_recvs =
                    ny * nz; // where we recieve from in each direction.
                int per_rank =
                    m_nlocal / n_recvs;  // how many per rank we have recieved
                int rank = i / per_rank; // which rank I am from

                int i_local =
                    i % per_rank; // my idx local to the rank I am from

                int3 local_coords;

                local_coords.x = i_local / (local_dims.y * local_dims.z);
                local_coords.y =
                    (i_local - local_coords.x * local_dims.y * local_dims.z) /
                    local_dims.z;
                local_coords.z =
                    (i_local - local_coords.x * local_dims.y * local_dims.z) -
                    local_coords.y * local_dims.z;

                int z_coord = rank / ny; // z is slow index for sends

                int y_coord = rank - z_coord * ny; // y is fast index for sends

                int z_offset = (m_local_grid_size[2] / nz) * z_coord;

                int y_offset = (m_local_grid_size[1] / ny) * y_coord;

                int3 global_coords =
                    make_int3(local_coords.x, local_coords.y + y_offset,
                              local_coords.z + z_offset);
                global_idx.x += global_coords.x;
                global_idx.y += global_coords.y;
                global_idx.z += global_coords.z;
            }

            return global_idx;
        }
    }
};

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
class Dfft {
  private:
    template <class T> inline void _forward(T* buff1, T* buff2);
    template <class T> inline void _backward(T* buff1, T* buff2);

  public:
    Dist<CollectiveComm, MPI_T, REORDER_T>& dist;
    FFTBackend FFTs;
    int ng[3];
    int nlocal;
    bool k_in_blocks;
    hqfftDist3d m_dist3d;

    Dfft(Dist<CollectiveComm, MPI_T, REORDER_T>& dist_, bool k_in_blocks_);
    ~Dfft();

    hqfftDist3d dist3d();
    int3 get_ks(int idx);
    int3 get_rs(int idx);

#ifdef SWFFT_GPU
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch);
    void forward(complexFloatDevice* data, complexFloatDevice* scratch);
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch);
    void backward(complexFloatDevice* data, complexFloatDevice* scratch);
#endif

    void forward(complexDoubleHost* data, complexDoubleHost* scratch);
    void forward(complexFloatHost* data, complexFloatHost* scratch);
    void backward(complexDoubleHost* data, complexDoubleHost* scratch);
    void backward(complexFloatHost* data, complexFloatHost* scratch);

    int buff_sz();

    int3 local_ng();
    int local_ng(int i);
};

} // namespace HQFFT
} // namespace SWFFT

#endif // SWFFT_HQFFT
#endif // _SWFFT_HQFFT_DFFT_HPP_