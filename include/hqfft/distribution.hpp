/**
 * @file hqfft/distribution.hpp
 */

#ifndef _SWFFT_HQFFT_DISTRIBUTION_HPP_
#define _SWFFT_HQFFT_DISTRIBUTION_HPP_
#ifdef SWFFT_HQFFT

#include "collectivecomm.hpp"
#include "common/copy_buffers.hpp"
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
template <template <class> class Communicator, class MPI_T, class REORDER_T>
class Distribution {
  private:
    template <class T> void _pencils_1(T* buff1, T* buff2);

    template <class T> void _inverse_pencils_1(T* buff1, T* buff2);

    template <class T> void _pencils_2(T* buff1, T* buff2);

    template <class T> void _inverse_pencils_2(T* buff1, T* buff2);

    template <class T> void _pencils_3(T* buff1, T* buff2);

    template <class T> void _inverse_pencils_3(T* buff1, T* buff2);

    template <class T> void _return_pencils(T* buff1, T* buff2);

  public:
    int ng[3];
    int nlocal;
    int world_size;
    int world_rank;
    int local_grid_size[3];
    int dims[3];
    int coords[3];
    int local_coords_start[3];
    MPI_Comm world_comm;
    MPI_Comm distcomms[4];

    Communicator<MPI_T> CollectiveComm;
    REORDER_T reorder;

    int blockSize;

    Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
    ~Distribution();

    void pencils_1(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void pencils_1(complexFloatHost* buff1, complexFloatHost* buff2);
    void pencils_2(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void pencils_2(complexFloatHost* buff1, complexFloatHost* buff2);
    void pencils_3(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void pencils_3(complexFloatHost* buff1, complexFloatHost* buff2);
    void inverse_pencils_1(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void inverse_pencils_1(complexFloatHost* buff1, complexFloatHost* buff2);
    void inverse_pencils_2(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void inverse_pencils_2(complexFloatHost* buff1, complexFloatHost* buff2);
    void inverse_pencils_3(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void inverse_pencils_3(complexFloatHost* buff1, complexFloatHost* buff2);
    void return_pencils(complexDoubleHost* buff1, complexDoubleHost* buff2);
    void return_pencils(complexFloatHost* buff1, complexFloatHost* buff2);

#ifdef SWFFT_GPU
    void pencils_1(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
    void pencils_1(complexFloatDevice* buff1, complexFloatDevice* buff2);
    void pencils_2(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
    void pencils_2(complexFloatDevice* buff1, complexFloatDevice* buff2);
    void pencils_3(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
    void pencils_3(complexFloatDevice* buff1, complexFloatDevice* buff2);
    void inverse_pencils_1(complexDoubleDevice* buff1,
                           complexDoubleDevice* buff2);
    void inverse_pencils_1(complexFloatDevice* buff1,
                           complexFloatDevice* buff2);
    void inverse_pencils_2(complexDoubleDevice* buff1,
                           complexDoubleDevice* buff2);
    void inverse_pencils_2(complexFloatDevice* buff1,
                           complexFloatDevice* buff2);
    void inverse_pencils_3(complexDoubleDevice* buff1,
                           complexDoubleDevice* buff2);
    void inverse_pencils_3(complexFloatDevice* buff1,
                           complexFloatDevice* buff2);
    void return_pencils(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
    void return_pencils(complexFloatDevice* buff1, complexFloatDevice* buff2);
#endif

    template <class T> void reshape_1(T* buff1, T* buff2);

    template <class T> void inverse_reshape_1(T* buff1, T* buff2);

    template <class T> void unreshape_1(T* buff1, T* buff2);

    template <class T> void inverse_unreshape_1(T* buff1, T* buff2);

    template <class T> void reshape_2(T* buff1, T* buff2);

    template <class T> void inverse_reshape_2(T* buff1, T* buff2);

    template <class T> void unreshape_2(T* buff1, T* buff2);

    template <class T> void inverse_unreshape_2(T* buff1, T* buff2);

    template <class T> void reshape_3(T* buff1, T* buff2);

    template <class T> void inverse_reshape_3(T* buff1, T* buff2);

    template <class T> void unreshape_3(T* buff1, T* buff2);

    template <class T> void inverse_unreshape_3(T* buff1, T* buff2);

    template <class T> void reshape_final(T* buff1, T* buff2, int ny, int nz);
};
} // namespace HQFFT
} // namespace SWFFT
#endif // SWFFT_HQFFT
#endif // _SWFFT_HQFFT_DISTRIBUTION_HPP_
