#ifdef SWFFT_HQFFT

#include "hqfft/dfft.hpp"

namespace SWFFT {
namespace HQFFT {

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::Dfft(
    Dist<CollectiveComm, MPI_T, REORDER_T>& dist_, bool k_in_blocks_)
    : dist(dist_), ng{dist.ng[0], dist.ng[1], dist.ng[2]}, nlocal(dist.nlocal),
      k_in_blocks(k_in_blocks_),
      m_dist3d(k_in_blocks, dist.local_grid_size, dist.local_coords_start,
               dist.nlocal, dist.ng, dist.dims, dist.coords) {}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::~Dfft() {}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
hqfftDist3d Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::dist3d() {
    return m_dist3d;
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int3 Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::get_ks(int idx) {
    return m_dist3d.get_ks(idx);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int3 Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::get_rs(int idx) {
    return m_dist3d.get_rs(idx);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
template <class T>
inline void
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::_forward(T* buff1,
                                                                   T* buff2) {

    dist.pencils_1(buff1, buff2);

    FFTs.forward(buff1, buff2, ng[2], nlocal / ng[2]);

    dist.pencils_2(buff2, buff1);

    FFTs.forward(buff1, buff2, ng[1], nlocal / ng[1]);

    dist.pencils_3(buff2, buff1);

    FFTs.forward(buff1, buff2, ng[0], nlocal / ng[0]);

    if (k_in_blocks) {

        dist.return_pencils(buff1, buff2);

    } else {

        copyBuffers<T> cpy(buff1, buff2, nlocal);
        cpy.wait();
    }
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
template <class T>
inline void
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::_backward(T* buff1,
                                                                    T* buff2) {
    if (k_in_blocks) {
        dist.pencils_1(buff1, buff2);

        FFTs.backward(buff1, buff2, ng[2], nlocal / ng[2]);

        dist.pencils_2(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[1], nlocal / ng[1]);

        dist.pencils_3(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[0], nlocal / ng[0]);

        dist.return_pencils(buff1, buff2);
    } else {

        FFTs.backward(buff1, buff2, ng[0], nlocal / ng[0]);

        dist.inverse_pencils_3(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[1], nlocal / ng[1]);

        dist.inverse_pencils_2(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[2], nlocal / ng[2]);

        dist.inverse_pencils_1(buff2, buff1);

        copyBuffers<T> cpy(buff1, buff2, nlocal);
        cpy.wait();
    }
}

#ifdef SWFFT_GPU
template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexDoubleDevice* buff1, complexDoubleDevice* buff2) {
    _forward(buff1, buff2);
    gpuDeviceSynchronize();
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexFloatDevice* buff1, complexFloatDevice* buff2) {
    _forward(buff1, buff2);
    gpuDeviceSynchronize();
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexDoubleDevice* buff1, complexDoubleDevice* buff2) {
    _backward(buff1, buff2);
    gpuDeviceSynchronize();
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexFloatDevice* buff1, complexFloatDevice* buff2) {
    _backward(buff1, buff2);
    gpuDeviceSynchronize();
}
#endif

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexDoubleHost* buff1, complexDoubleHost* buff2) {
    _forward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexFloatHost* buff1, complexFloatHost* buff2) {
    _forward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexDoubleHost* buff1, complexDoubleHost* buff2) {
    _backward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexFloatHost* buff1, complexFloatHost* buff2) {
    _backward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::buff_sz() {
    return nlocal;
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int3 Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::local_ng() {
    return make_int3(dist.local_grid_size[0], dist.local_grid_size[1],
                     dist.local_grid_size[2]);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::local_ng(int i) {
    return dist.local_grid_size[i];
}

#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
template class Dfft<Distribution, GPUReshape, AllToAll, CPUMPI, gpuFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, CPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, AllToAll, CPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, CPUMPI, gpuFFT>;
#endif
#ifdef SWFFT_FFTW
template class Dfft<Distribution, GPUReshape, AllToAll, CPUMPI, fftw>;
template class Dfft<Distribution, GPUReshape, PairSends, CPUMPI, fftw>;
#endif
template class Dfft<Distribution, GPUReshape, AllToAll, CPUMPI, TestFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, CPUMPI, TestFFT>;
#endif

#ifdef SWFFT_FFTW
template class Dfft<Distribution, CPUReshape, AllToAll, CPUMPI, fftw>;
template class Dfft<Distribution, CPUReshape, PairSends, CPUMPI, fftw>;
#endif

template class Dfft<Distribution, CPUReshape, AllToAll, CPUMPI, TestFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, CPUMPI, TestFFT>;

#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI
#ifdef SWFFT_CUFFT
template class Dfft<Distribution, GPUReshape, AllToAll, GPUMPI, gpuFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, GPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, AllToAll, GPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, GPUMPI, gpuFFT>;
#endif
#ifdef SWFFT_FFTW
template class Dfft<Distribution, GPUReshape, AllToAll, GPUMPI, fftw>;
template class Dfft<Distribution, GPUReshape, PairSends, GPUMPI, fftw>;
#endif
template class Dfft<Distribution, GPUReshape, AllToAll, GPUMPI, TestFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, GPUMPI, TestFFT>;

#ifdef SWFFT_FFTW
template class Dfft<Distribution, CPUReshape, AllToAll, GPUMPI, fftw>;
template class Dfft<Distribution, CPUReshape, PairSends, GPUMPI, fftw>;
#endif

template class Dfft<Distribution, CPUReshape, AllToAll, GPUMPI, TestFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, GPUMPI, TestFFT>;
#endif
#endif

} // namespace HQFFT
} // namespace SWFFT
#endif // SWFFT_HQFFT