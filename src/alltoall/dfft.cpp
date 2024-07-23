#ifdef SWFFT_ALLTOALL
#include "alltoall/dfft.hpp"

namespace SWFFT {

namespace A2A {
template <class MPI_T, class REORDER_T, class FFTBackend>
Dfft<MPI_T, REORDER_T, FFTBackend>::Dfft(Distribution<MPI_T, REORDER_T>& dist_,
                                         bool ks_as_block_)
    : ks_as_block(ks_as_block_), dist(dist_),
      m_dist3d(ks_as_block, dist.local_grid_size, dist.local_coordinates_start,
               dist.nlocal, dist.world_size, dist.dims, dist.fftcomms[1]) {
    ng[0] = dist.ng[0];
    ng[1] = dist.ng[1];
    ng[2] = dist.ng[2];

    nlocal = dist.nlocal;

    world_size = dist.world_size;
    world_rank = dist.world_rank;

    blockSize = dist.blockSize;
}

template <class MPI_T, class REORDER_T, class FFTBackend>
Dfft<MPI_T, REORDER_T, FFTBackend>::~Dfft() {}

template <class MPI_T, class REORDER_T, class FFTBackend>
alltoallDist3d Dfft<MPI_T, REORDER_T, FFTBackend>::dist3d() {
    return m_dist3d;
}

template <class MPI_T, class REORDER_T, class FFTBackend>
int3 Dfft<MPI_T, REORDER_T, FFTBackend>::get_ks(int idx) {
    return m_dist3d.get_ks(idx);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
int3 Dfft<MPI_T, REORDER_T, FFTBackend>::get_rs(int idx) {
    return m_dist3d.get_rs(idx);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
template <class T>
void Dfft<MPI_T, REORDER_T, FFTBackend>::fft(T* data, T* scratch,
                                             fftdirection direction) {
    if (ks_as_block) {
#pragma GCC unroll 3
        for (int i = 0; i < 3; i++) {
            dist.get_pencils(data, scratch, i);
            dist.reorder(data, scratch, i, 0);

            int dim = (i + 2) % 3;

            int nFFTs = (nlocal / ng[dim]);
            if (direction == FFT_FORWARD) {
                FFTs.forward(data, scratch, ng[dim], nFFTs);
            } else {
                FFTs.backward(data, scratch, ng[dim], nFFTs);
            }

            dist.reorder(data, scratch, i, 1);
            dist.return_pencils(data, scratch, i);
            dist.shuffle_indices(data, scratch, i);
        }
    } else {

        if (direction == FFT_FORWARD) {
            for (int i = 0; i < 2; i++) {

                int dim = (i + 2) % 3;

                dist.get_pencils(data, scratch, i);
                dist.reorder(data, scratch, i, 0);

                int nFFTs = (nlocal / ng[dim]);
                FFTs.forward(data, scratch, ng[dim], nFFTs);

                dist.reorder(data, scratch, i, 1);
                dist.return_pencils(data, scratch, i);
                dist.shuffle_indices(data, scratch, i);
            }
            dist.get_pencils(data, scratch, 2);
            dist.reorder(data, scratch, 2, 0);
            int dim = (2 + 2) % 3;
            int nFFTs = (nlocal / ng[dim]);
            FFTs.forward(data, scratch, ng[dim], nFFTs);
            dist.copy(data, scratch);

        } else {
            int dim = (2 + 2) % 3;
            int nFFTs = (nlocal / ng[dim]);
            FFTs.backward(data, scratch, ng[dim], nFFTs);

            dist.reorder(data, scratch, 2, 1);
            dist.return_pencils(data, scratch, 2);
            dist.shuffle_indices(data, scratch, 2);

            dist.get_pencils(data, scratch, 0);
            dist.reorder(data, scratch, 0, 0);

            dim = (0 + 2) % 3;
            nFFTs = (nlocal / ng[dim]);
            FFTs.backward(data, scratch, ng[dim], nFFTs);

            dist.reorder(data, scratch, 0, 1);
            dist.return_pencils(data, scratch, 0);
            dist.shuffle_indices(data, scratch, 0);

            dist.get_pencils(data, scratch, 1);
            dist.reorder(data, scratch, 1, 0);

            dim = (1 + 2) % 3;
            nFFTs = (nlocal / ng[dim]);
            FFTs.backward(data, scratch, ng[dim], nFFTs);

            dist.reorder(data, scratch, 1, 1);
            dist.return_pencils(data, scratch, 1);
            dist.shuffle_indices(data, scratch, 3);
        }
    }
}

#ifdef SWFFT_GPU
template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::forward(complexDoubleDevice* data,
                                                 complexDoubleDevice* scratch) {
    fft(data, scratch, FFT_FORWARD);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::forward(complexFloatDevice* data,
                                                 complexFloatDevice* scratch) {
    fft(data, scratch, FFT_FORWARD);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::backward(
    complexDoubleDevice* data, complexDoubleDevice* scratch) {
    fft(data, scratch, FFT_BACKWARD);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::backward(complexFloatDevice* data,
                                                  complexFloatDevice* scratch) {
    fft(data, scratch, FFT_BACKWARD);
}
#endif

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::forward(complexDoubleHost* data,
                                                 complexDoubleHost* scratch) {
    fft(data, scratch, FFT_FORWARD);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::forward(complexFloatHost* data,
                                                 complexFloatHost* scratch) {
    fft(data, scratch, FFT_FORWARD);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::backward(complexDoubleHost* data,
                                                  complexDoubleHost* scratch) {
    fft(data, scratch, FFT_BACKWARD);
}

template <class MPI_T, class REORDER_T, class FFTBackend>
void Dfft<MPI_T, REORDER_T, FFTBackend>::backward(complexFloatHost* data,
                                                  complexFloatHost* scratch) {
    fft(data, scratch, FFT_BACKWARD);
}

} // namespace A2A
} // namespace SWFFT

#ifdef SWFFT_FFTW
template class SWFFT::A2A::Dfft<SWFFT::CPUMPI, SWFFT::A2A::CPUReorder,
                                SWFFT::FFTWPlanManager>;
#ifdef SWFFT_GPU
template class SWFFT::A2A::Dfft<SWFFT::CPUMPI, SWFFT::A2A::GPUReorder,
                                SWFFT::FFTWPlanManager>;
#ifndef SWFFT_NOCUDAMPI
template class SWFFT::A2A::Dfft<SWFFT::GPUMPI, SWFFT::A2A::CPUReorder,
                                SWFFT::FFTWPlanManager>;
template class SWFFT::A2A::Dfft<SWFFT::GPUMPI, SWFFT::A2A::GPUReorder,
                                SWFFT::FFTWPlanManager>;
#endif
#endif
#endif

template class SWFFT::A2A::Dfft<SWFFT::CPUMPI, SWFFT::A2A::CPUReorder,
                                SWFFT::TestFFT>;
#ifdef SWFFT_GPU
template class SWFFT::A2A::Dfft<SWFFT::CPUMPI, SWFFT::A2A::GPUReorder,
                                SWFFT::TestFFT>;
#ifdef SWFFT_CUFFT
template class SWFFT::A2A::Dfft<SWFFT::CPUMPI, SWFFT::A2A::CPUReorder,
                                SWFFT::GPUPlanManager>;
template class SWFFT::A2A::Dfft<SWFFT::CPUMPI, SWFFT::A2A::GPUReorder,
                                SWFFT::GPUPlanManager>;
#ifndef SWFFT_NOCUDAMPI
template class SWFFT::A2A::Dfft<SWFFT::GPUMPI, SWFFT::A2A::CPUReorder,
                                SWFFT::GPUPlanManager>;
template class SWFFT::A2A::Dfft<SWFFT::GPUMPI, SWFFT::A2A::GPUReorder,
                                SWFFT::GPUPlanManager>;
template class SWFFT::A2A::Dfft<SWFFT::GPUMPI, SWFFT::A2A::GPUReorder,
                                SWFFT::TestFFT>;
template class SWFFT::A2A::Dfft<SWFFT::GPUMPI, SWFFT::A2A::CPUReorder,
                                SWFFT::TestFFT>;
#endif
#endif
#endif

#endif