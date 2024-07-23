/**
 * @file hqfft/dfft.hpp
 */

#ifndef _SWFFT_HQFFT_DFFT_HPP_
#define _SWFFT_HQFFT_DFFT_HPP_
#ifdef SWFFT_HQFFT

#include "common/copy_buffers.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "mpi/mpi_isend_irecv.hpp"
#include "mpi/mpiwrangler.hpp"
#include "collectivecomm.hpp"
#include "distribution.hpp"
#include "hqfft_reorder.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>
#include <stdio.h>

namespace SWFFT {
namespace HQFFT {

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

    Dfft(Dist<CollectiveComm, MPI_T, REORDER_T>& dist_, bool k_in_blocks_);
    ~Dfft();

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