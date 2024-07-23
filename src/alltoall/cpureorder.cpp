#ifdef SWFFT_ALLTOALL
#include "alltoall/reorder.hpp"
namespace SWFFT {
namespace A2A {
namespace CPUREORDER {
template <class T>
void reorder_forwards_kernel(const T* __restrict src, T* __restrict dest,
                             int mini_pencils_per_rank, int world_size,
                             int mini_pencil_size, int n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        int new_idx = calc_mini_pencil_idx(i, mini_pencils_per_rank, world_size,
                                           mini_pencil_size);
        dest[new_idx] = src[i];
    }
}

template <class T>
void reorder_backwards_kernel(const T* __restrict src, T* __restrict dest,
                              int mini_pencils_per_rank, int world_size,
                              int mini_pencil_size, int n) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < n; i++) {
        int new_idx = calc_mini_pencil_idx(i, mini_pencils_per_rank, world_size,
                                           mini_pencil_size);
        dest[i] = src[new_idx];
    }
}

template <class T>
void d_fast_z_to_x(const T* __restrict source, T* __restrict dest, int lgridx,
                   int lgridy, int lgridz, int nlocal) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < nlocal; idx++) {

        int i = idx / (lgridx * lgridy);
        int j = (idx - (i * lgridx * lgridy)) / lgridx;
        int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

        int dest_index = i * lgridy * lgridx + j * lgridx + k;
        int source_index = k * lgridy * lgridz + j * lgridz + i;

        dest[dest_index] = source[source_index];
    }
}

template <class T>
void d_fast_x_to_z(const T* __restrict source, T* __restrict dest, int lgridx,
                   int lgridy, int lgridz, int nlocal) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < nlocal; idx++) {

        int i = idx / (lgridx * lgridy);
        int j = (idx - (i * lgridx * lgridy)) / lgridx;
        int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

        int dest_index = i * lgridy * lgridx + j * lgridx + k;
        int source_index = k * lgridy * lgridz + j * lgridz + i;

        dest[source_index] = source[dest_index];
    }
}

template <class T>
void d_fast_x_to_y(const T* __restrict source, T* __restrict dest, int lgridx,
                   int lgridy, int lgridz, int nlocal) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < nlocal; idx++) {

        int i = idx / (lgridz * lgridy);
        int j = (idx - (i * lgridz * lgridy)) / lgridy;
        int k = idx - (i * (lgridz * lgridy)) - (j * lgridy);

        int dest_index = i * lgridz * lgridy + j * lgridy + k;
        int source_index = j * lgridx * lgridy + k * lgridx + i;

        dest[dest_index] = source[source_index];
    }
}

template <class T>
void d_fast_y_to_z(const T* __restrict source, T* __restrict dest, int lgridx,
                   int lgridy, int lgridz, int nlocal) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int idx = 0; idx < nlocal; idx++) {

        int i = idx / (lgridz * lgridy);
        int j = (idx - (i * lgridz * lgridy)) / lgridz;
        int k = idx - (i * (lgridz * lgridy)) - (j * lgridz);

        int dest_index = i * lgridz * lgridy + j * lgridz + k;
        int source_index = i * lgridz * lgridy + k * lgridy + j;

        dest[dest_index] = source[source_index];
    }
}
} // namespace CPUREORDER

CPUReorder::CPUReorder(int3 ng_, int3 dims_, int3 coords_, int blockSize_)
    : m_ng(ng_), m_dims(dims_), m_coords(coords_), m_blockSize(blockSize_) {
    m_local_grid.x = m_ng.x / m_dims.x;
    m_local_grid.y = m_ng.y / m_dims.y;
    m_local_grid.z = m_ng.z / m_dims.z;
    m_nlocal = m_local_grid.x * m_local_grid.y * m_local_grid.z;
    m_local_grid_size[0] = m_local_grid.x;
    m_local_grid_size[1] = m_local_grid.y;
    m_local_grid_size[2] = m_local_grid.z;
    m_world_size = m_dims.x * m_dims.y * m_dims.z;
}
CPUReorder::~CPUReorder(){};
CPUReorder::CPUReorder(){};

template <class T>
void CPUReorder::cpu_shuffle_indices(T* Buff1, T* Buff2, int n) {
    switch (n) {
    case 0:
        CPUREORDER::d_fast_z_to_x(Buff2, Buff1, m_local_grid.x, m_local_grid.y,
                                  m_local_grid.z, m_nlocal);
        break;
    case 1:
        CPUREORDER::d_fast_x_to_y(Buff2, Buff1, m_local_grid.x, m_local_grid.y,
                                  m_local_grid.z, m_nlocal);
        break;
    case 2:
        CPUREORDER::d_fast_y_to_z(Buff2, Buff1, m_local_grid.x, m_local_grid.y,
                                  m_local_grid.z, m_nlocal);
        break;
    case 3:
        CPUREORDER::d_fast_x_to_z(Buff2, Buff1, m_local_grid.x, m_local_grid.y,
                                  m_local_grid.z, m_nlocal);
        break;
    }
}

void CPUReorder::shuffle_indices(complexDoubleHost* Buff1,
                                 complexDoubleHost* Buff2, int n) {
    cpu_shuffle_indices(Buff1, Buff2, n);
}

void CPUReorder::shuffle_indices(complexFloatHost* Buff1,
                                 complexFloatHost* Buff2, int n) {
    cpu_shuffle_indices(Buff1, Buff2, n);
}

#ifdef SWFFT_GPU
void CPUReorder::shuffle_indices(complexDoubleDevice* Buff1,
                                 complexDoubleDevice* Buff2, int n) {
    complexDoubleHost* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexDoubleDevice));
    complexDoubleHost* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexDoubleDevice));
    gpuMemcpy(d_buff1, Buff1, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyDeviceToHost);
    cpu_shuffle_indices(d_buff1, d_buff2, n);
    gpuMemcpy(Buff2, d_buff2, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyHostToDevice);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void CPUReorder::shuffle_indices(complexFloatDevice* Buff1,
                                 complexFloatDevice* Buff2, int n) {
    complexFloatHost* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexFloatHost));
    complexFloatHost* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexFloatHost));
    gpuMemcpy(d_buff1, Buff1, m_nlocal * sizeof(complexFloatDevice),
              cudaMemcpyDeviceToHost);
    cpu_shuffle_indices(d_buff1, d_buff2, n);
    gpuMemcpy(Buff2, d_buff2, m_nlocal * sizeof(complexFloatDevice),
              cudaMemcpyHostToDevice);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}
#endif

template <class T>
void CPUReorder::cpu_reorder(T* Buff1, T* Buff2, int n, int direction) {
    int dim = (n + 2) % 3;
    int nsends = (m_nlocal / m_world_size);
    if (direction == 0) {
        CPUREORDER::reorder_forwards_kernel(
            Buff2, Buff1, nsends / m_local_grid_size[dim], m_world_size,
            m_local_grid_size[dim], m_nlocal);
    } else {
        CPUREORDER::reorder_backwards_kernel(
            Buff2, Buff1, nsends / m_local_grid_size[dim], m_world_size,
            m_local_grid_size[dim], m_nlocal);
    }
}

void CPUReorder::reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                         int n, int direction) {
    cpu_reorder(Buff1, Buff2, n, direction);
}

void CPUReorder::reorder(complexFloatHost* Buff1, complexFloatHost* Buff2,
                         int n, int direction) {
    cpu_reorder(Buff1, Buff2, n, direction);
}

#ifdef SWFFT_GPU
void CPUReorder::reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                         int n, int direction) {
    complexDoubleHost* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexDoubleDevice));
    complexDoubleHost* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexDoubleDevice));
    gpuMemcpy(d_buff1, Buff1, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyDeviceToHost);
    cpu_reorder(d_buff1, d_buff2, n, direction);
    gpuMemcpy(Buff2, d_buff2, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyHostToDevice);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void CPUReorder::reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                         int n, int direction) {
    complexFloatHost* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexFloatDevice));
    complexFloatHost* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexFloatDevice));
    gpuMemcpy(d_buff1, Buff1, m_nlocal * sizeof(complexFloatHost),
              cudaMemcpyDeviceToHost);
    cpu_reorder(d_buff1, d_buff2, n, direction);
    gpuMemcpy(Buff2, d_buff2, m_nlocal * sizeof(complexFloatHost),
              cudaMemcpyHostToDevice);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}
#endif // SWFFT_GPU

} // namespace A2A
} // namespace SWFFT

#endif // SWFFT_ALLTOALL
