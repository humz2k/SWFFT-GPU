#ifdef SWFFT_PAIRWISE

#define PENCIL

#include "pairwise/dfft.hpp"
#include <cassert>

#ifndef USE_SLAB_WORKAROUND
#define USE_SLAB_WORKAROUND 0
#endif

#define DEBUG_CONDITION false
namespace SWFFT {
namespace PAIR {

template <class MPI_T, class FFTBackend>
Dfft<MPI_T, FFTBackend>::Dfft(MPI_Comm comm_, int nx, int ny, int nz)
    : m_comm(comm_), m_n{nx, ny, nz},
      m_double_dist(comm_, nx, ny, nz, DEBUG_CONDITION),
      m_float_dist(comm_, nx, ny, nz, DEBUG_CONDITION),
      m_dist3d(m_double_dist.process_topology_2_z,
               m_double_dist.process_topology_3) {}

template <class MPI_T, class FFTBackend>
int3 Dfft<MPI_T, FFTBackend>::coords() {
    return make_int3(m_double_dist.get_self_3d(0), m_double_dist.get_self_3d(1),
                     m_double_dist.get_self_3d(2));
}

template <class MPI_T, class FFTBackend>
pairwiseDist3d Dfft<MPI_T, FFTBackend>::dist3d() {
    return m_dist3d;
}

template <class MPI_T, class FFTBackend>
int3 Dfft<MPI_T, FFTBackend>::get_ks(int idx) {
    return m_dist3d.get_ks(idx);
}

template <class MPI_T, class FFTBackend>
int3 Dfft<MPI_T, FFTBackend>::get_rs(int idx) {
    return m_dist3d.get_rs(idx);
}

template <class MPI_T, class FFTBackend>
int Dfft<MPI_T, FFTBackend>::get_local_ng_3d(int direction) {
    return m_double_dist.local_ng_3d(direction);
}

template <class MPI_T, class FFTBackend> Dfft<MPI_T, FFTBackend>::~Dfft() {}

template <class MPI_T, class FFTBackend>
template <class T>
void Dfft<MPI_T, FFTBackend>::_forward(T* data) {
    T* scratch;
    swfftAlloc(&scratch, sizeof(T) * buff_sz());
    forward(data, scratch);
    swfftFree(scratch);
}

template <class MPI_T, class FFTBackend>
template <class T>
void Dfft<MPI_T, FFTBackend>::_backward(T* data) {
    T* scratch;
    swfftAlloc(&scratch, sizeof(T) * buff_sz());
    backward(data, scratch);
    swfftFree(scratch);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexDoubleHost* data) {
    _forward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexFloatHost* data) {
    _forward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexDoubleHost* data) {
    _backward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexFloatHost* data) {
    _backward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexDoubleHost* data,
                                      complexDoubleHost* scratch) {
    m_double_dist.dist_3_to_2(data, scratch, 0);

    m_FFTs.forward(scratch, data, m_double_dist.process_topology_2_x.n[0],
                   m_double_dist.process_topology_2_x.n[1] *
                       m_double_dist.process_topology_2_x.n[2]);

    m_double_dist.dist_2_to_3(data, scratch, 0);

    m_double_dist.dist_3_to_2(scratch, data, 1);

    m_FFTs.forward(data, scratch, m_double_dist.process_topology_2_y.n[1],
                   m_double_dist.process_topology_2_y.n[0] *
                       m_double_dist.process_topology_2_y.n[2]);

    m_double_dist.dist_2_to_3(scratch, data, 1);

    m_double_dist.dist_3_to_2(data, scratch, 2);

    m_FFTs.forward(scratch, data, m_double_dist.process_topology_2_z.n[2],
                   m_double_dist.process_topology_2_z.n[0] *
                       m_double_dist.process_topology_2_z.n[1]);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexFloatHost* data,
                                      complexFloatHost* scratch) {
    m_float_dist.dist_3_to_2(data, scratch, 0);

    m_FFTs.forward(scratch, data, m_float_dist.process_topology_2_x.n[0],
                   m_float_dist.process_topology_2_x.n[1] *
                       m_float_dist.process_topology_2_x.n[2]);

    m_float_dist.dist_2_to_3(data, scratch, 0);

    m_float_dist.dist_3_to_2(scratch, data, 1);

    m_FFTs.forward(data, scratch, m_float_dist.process_topology_2_y.n[1],
                   m_float_dist.process_topology_2_y.n[0] *
                       m_float_dist.process_topology_2_y.n[2]);

    m_float_dist.dist_2_to_3(scratch, data, 1);

    m_float_dist.dist_3_to_2(data, scratch, 2);

    m_FFTs.forward(scratch, data, m_float_dist.process_topology_2_z.n[2],
                   m_float_dist.process_topology_2_z.n[0] *
                       m_float_dist.process_topology_2_z.n[1]);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexDoubleHost* data,
                                       complexDoubleHost* scratch) {

    m_FFTs.backward(data, scratch, m_double_dist.process_topology_2_z.n[2],
                    m_double_dist.process_topology_2_z.n[0] *
                        m_double_dist.process_topology_2_z.n[1]);

    m_double_dist.dist_2_to_3(scratch, data, 2);

    m_double_dist.dist_3_to_2(data, scratch, 1);

    m_FFTs.backward(scratch, data, m_double_dist.process_topology_2_y.n[1],
                    m_double_dist.process_topology_2_y.n[0] *
                        m_double_dist.process_topology_2_y.n[2]);

    m_double_dist.dist_2_to_3(data, scratch, 1);

    m_double_dist.dist_3_to_2(scratch, data, 0);

    m_FFTs.backward(data, scratch, m_double_dist.process_topology_2_x.n[0],
                    m_double_dist.process_topology_2_x.n[1] *
                        m_double_dist.process_topology_2_x.n[2]);

    m_double_dist.dist_2_to_3(scratch, data, 0);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexFloatHost* data,
                                       complexFloatHost* scratch) {

    m_FFTs.backward(data, scratch, m_float_dist.process_topology_2_z.n[2],
                    m_float_dist.process_topology_2_z.n[0] *
                        m_float_dist.process_topology_2_z.n[1]);

    m_float_dist.dist_2_to_3(scratch, data, 2);

    m_float_dist.dist_3_to_2(data, scratch, 1);

    m_FFTs.backward(scratch, data, m_float_dist.process_topology_2_y.n[1],
                    m_float_dist.process_topology_2_y.n[0] *
                        m_float_dist.process_topology_2_y.n[2]);

    m_float_dist.dist_2_to_3(data, scratch, 1);

    m_float_dist.dist_3_to_2(scratch, data, 0);

    m_FFTs.backward(data, scratch, m_float_dist.process_topology_2_x.n[0],
                    m_float_dist.process_topology_2_x.n[1] *
                        m_float_dist.process_topology_2_x.n[2]);

    m_float_dist.dist_2_to_3(scratch, data, 0);
}

#ifdef SWFFT_GPU
template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexDoubleDevice* data) {
    size_t sz = sizeof(complexDoubleDevice) * buff_sz();
    complexDoubleHost* h_data;
    swfftAlloc(&h_data, sz);
    gpuMemcpy(h_data, data, sz, gpuMemcpyDeviceToHost);
    forward(h_data);
    gpuMemcpy(data, h_data, sz, gpuMemcpyHostToDevice);
    swfftFree(h_data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexFloatDevice* data) {
    size_t sz = sizeof(complexFloatDevice) * buff_sz();
    complexFloatHost* h_data;
    swfftAlloc(&h_data, sz);
    gpuMemcpy(h_data, data, sz, gpuMemcpyDeviceToHost);
    forward(h_data);
    gpuMemcpy(data, h_data, sz, gpuMemcpyHostToDevice);
    swfftFree(h_data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexDoubleDevice* data) {
    size_t sz = sizeof(complexDoubleDevice) * buff_sz();
    complexDoubleHost* h_data;
    swfftAlloc(&h_data, sz);
    gpuMemcpy(h_data, data, sz, gpuMemcpyDeviceToHost);
    backward(h_data);
    gpuMemcpy(data, h_data, sz, gpuMemcpyHostToDevice);
    swfftFree(h_data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexFloatDevice* data) {
    size_t sz = sizeof(complexFloatDevice) * buff_sz();
    complexFloatHost* h_data;
    swfftAlloc(&h_data, sz);
    gpuMemcpy(h_data, data, sz, gpuMemcpyDeviceToHost);
    backward(h_data);
    gpuMemcpy(data, h_data, sz, gpuMemcpyHostToDevice);
    swfftFree(h_data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexDoubleDevice* data,
                                      complexDoubleDevice* scratch) {
    forward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::forward(complexFloatDevice* data,
                                      complexFloatDevice* scratch) {
    forward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexDoubleDevice* data,
                                       complexDoubleDevice* scratch) {
    backward(data);
}

template <class MPI_T, class FFTBackend>
void Dfft<MPI_T, FFTBackend>::backward(complexFloatDevice* data,
                                       complexFloatDevice* scratch) {
    backward(data);
}

#endif

template <class MPI_T, class FFTBackend>
int Dfft<MPI_T, FFTBackend>::get_nproc_3d(int direction) {
    return m_double_dist.get_nproc_3d(direction);
}

template <class MPI_T, class FFTBackend>
int Dfft<MPI_T, FFTBackend>::buff_sz() {
    int size = 1;
    for (int i = 0; i < 3; i++) {
        size *= m_n[i] / m_double_dist.process_topology_3.nproc[i];
    }
    return size;
}

template class Dfft<CPUMPI, TestFFT>;
#ifdef SWFFT_FFTW
template class Dfft<CPUMPI, fftw>;
#endif
#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
template class Dfft<CPUMPI, gpuFFT>;
#endif
#ifndef SWFFT_NOCUDAMPI
template class Dfft<GPUMPI, gpuFFT>;
template class Dfft<GPUMPI, fftw>;
template class Dfft<GPUMPI, TestFFT>;
#endif
#endif

} // namespace PAIR
} // namespace SWFFT

#endif