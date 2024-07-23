#ifdef SWFFT_ALLTOALL
#ifdef SWFFT_GPU
#include "alltoall/reorder.hpp"
namespace SWFFT {
namespace A2A {
namespace GPUREORDER {
template <class T>
__global__ void
reorder_forwards_kernel(const T* __restrict src, T* __restrict dest,
                        int mini_pencils_per_rank, int world_size,
                        int mini_pencil_size, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    int new_idx = calc_mini_pencil_idx(i, mini_pencils_per_rank, world_size,
                                       mini_pencil_size);
    dest[new_idx] = __ldg(&src[i]);
}

template <class T>
__global__ void
reorder_backwards_kernel(const T* __restrict src, T* __restrict dest,
                         int mini_pencils_per_rank, int world_size,
                         int mini_pencil_size, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    int new_idx = calc_mini_pencil_idx(i, mini_pencils_per_rank, world_size,
                                       mini_pencil_size);
    dest[i] = __ldg(&src[new_idx]);
}

template <class T>
__global__ void d_fast_z_to_x(const T* __restrict source, T* __restrict dest,
                              int lgridx, int lgridy, int lgridz, int nlocal) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal) {

        int i = idx / (lgridx * lgridy);
        int j = (idx - (i * lgridx * lgridy)) / lgridx;
        int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

        int dest_index = i * lgridy * lgridx + j * lgridx + k;
        int source_index = k * lgridy * lgridz + j * lgridz + i;

        dest[dest_index] = __ldg(&source[source_index]);
        // dest[dest_index * 2 + 1] = source[source_index * 2 + 1];
    }
}

template <class T>
__global__ void d_fast_x_to_z(const T* __restrict source, T* __restrict dest,
                              int lgridx, int lgridy, int lgridz, int nlocal) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal) {

        int i = idx / (lgridx * lgridy);
        int j = (idx - (i * lgridx * lgridy)) / lgridx;
        int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

        int dest_index = i * lgridy * lgridx + j * lgridx + k;
        int source_index = k * lgridy * lgridz + j * lgridz + i;

        dest[source_index] = __ldg(&source[dest_index]);
        // dest[dest_index * 2 + 1] = source[source_index * 2 + 1];
    }
}

template <class T>
__global__ void d_fast_x_to_y(const T* __restrict source, T* __restrict dest,
                              int lgridx, int lgridy, int lgridz, int nlocal) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal) {

        int i = idx / (lgridz * lgridy);
        int j = (idx - (i * lgridz * lgridy)) / lgridy;
        int k = idx - (i * (lgridz * lgridy)) - (j * lgridy);

        int dest_index = i * lgridz * lgridy + j * lgridy + k;
        int source_index = j * lgridx * lgridy + k * lgridx + i;

        dest[dest_index] = __ldg(&source[source_index]);
        // dest[dest_index * 2 + 1] = source[source_index * 2 + 1];
    }
}

template <class T>
__global__ void d_fast_y_to_z(const T* __restrict source, T* __restrict dest,
                              int lgridx, int lgridy, int lgridz, int nlocal) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal) {

        int i = idx / (lgridz * lgridy);
        int j = (idx - (i * lgridz * lgridy)) / lgridz;
        int k = idx - (i * (lgridz * lgridy)) - (j * lgridz);

        int dest_index = i * lgridz * lgridy + j * lgridz + k;
        int source_index = i * lgridz * lgridy + k * lgridy + j;

        dest[dest_index] = __ldg(&source[source_index]);
        // dest[dest_index * 2 + 1] = source[source_index * 2 + 1];
    }
}
} // namespace GPUREORDER

GPUReorder::GPUReorder(int3 ng, int3 dims, int3 coords, int blockSize)
    : m_ng(ng), m_dims(dims), m_coords(coords), m_blockSize(blockSize) {
    m_local_grid.x = m_ng.x / m_dims.x;
    m_local_grid.y = m_ng.y / m_dims.y;
    m_local_grid.z = m_ng.z / m_dims.z;
    m_nlocal = m_local_grid.x * m_local_grid.y * m_local_grid.z;
    m_local_grid_size[0] = m_local_grid.x;
    m_local_grid_size[1] = m_local_grid.y;
    m_local_grid_size[2] = m_local_grid.z;
    m_world_size = m_dims.x * m_dims.y * m_dims.z;
}
GPUReorder::~GPUReorder(){};
GPUReorder::GPUReorder(){};

template <class T>
inline void GPUReorder::gpu_shuffle_indices(T* Buff1, T* Buff2, int n) {
    int numBlocks = (m_nlocal + (m_blockSize - 1)) / m_blockSize;
    switch (n) {
    case 0:
        gpuLaunch(GPUREORDER::d_fast_z_to_x, numBlocks, m_blockSize, Buff2,
                  Buff1, m_local_grid.x, m_local_grid.y, m_local_grid.z,
                  m_nlocal);
        break;
    case 1:
        gpuLaunch(GPUREORDER::d_fast_x_to_y, numBlocks, m_blockSize, Buff2,
                  Buff1, m_local_grid.x, m_local_grid.y, m_local_grid.z,
                  m_nlocal);
        break;
    case 2:
        gpuLaunch(GPUREORDER::d_fast_y_to_z, numBlocks, m_blockSize, Buff2,
                  Buff1, m_local_grid.x, m_local_grid.y, m_local_grid.z,
                  m_nlocal);
        break;
    case 3:
        gpuLaunch(GPUREORDER::d_fast_x_to_z, numBlocks, m_blockSize, Buff2,
                  Buff1, m_local_grid.x, m_local_grid.y, m_local_grid.z,
                  m_nlocal);
        break;
    default:
        break;
    }
}

void GPUReorder::shuffle_indices(complexDoubleDevice* Buff1,
                                 complexDoubleDevice* Buff2, int n) {
    gpu_shuffle_indices(Buff1, Buff2, n);
}

void GPUReorder::shuffle_indices(complexFloatDevice* Buff1,
                                 complexFloatDevice* Buff2, int n) {
    gpu_shuffle_indices(Buff1, Buff2, n);
}

void GPUReorder::shuffle_indices(complexDoubleHost* Buff1,
                                 complexDoubleHost* Buff2, int n) {
    complexDoubleDevice* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexDoubleDevice));
    complexDoubleDevice* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexDoubleDevice));
    gpuMemcpy(d_buff2, Buff2, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyHostToDevice);
    gpu_shuffle_indices(d_buff1, d_buff2, n);
    gpuMemcpy(Buff1, d_buff1, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void GPUReorder::shuffle_indices(complexFloatHost* Buff1,
                                 complexFloatHost* Buff2, int n) {
    complexFloatDevice* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexFloatHost));
    complexFloatDevice* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexFloatHost));
    gpuMemcpy(d_buff2, Buff2, m_nlocal * sizeof(complexFloatDevice),
              cudaMemcpyHostToDevice);
    gpu_shuffle_indices(d_buff1, d_buff2, n);
    gpuMemcpy(Buff1, d_buff1, m_nlocal * sizeof(complexFloatDevice),
              cudaMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

template <class T>
inline void GPUReorder::gpu_reorder(T* Buff1, T* Buff2, int n, int direction) {
    int dim = (n + 2) % 3;
    int nsends = (m_nlocal / m_world_size);
    int numBlocks = (m_nlocal + (m_blockSize - 1)) / m_blockSize;
    if (direction == 0) {
        gpuLaunch(GPUREORDER::reorder_forwards_kernel, numBlocks, m_blockSize,
                  Buff2, Buff1, nsends / m_local_grid_size[dim], m_world_size,
                  m_local_grid_size[dim], m_nlocal);
    } else {
        gpuLaunch(GPUREORDER::reorder_backwards_kernel, numBlocks, m_blockSize,
                  Buff2, Buff1, nsends / m_local_grid_size[dim], m_world_size,
                  m_local_grid_size[dim], m_nlocal);
    }
}

void GPUReorder::reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2,
                         int n, int direction) {
    gpu_reorder(Buff1, Buff2, n, direction);
}

void GPUReorder::reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2,
                         int n, int direction) {
    gpu_reorder(Buff1, Buff2, n, direction);
}

void GPUReorder::reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2,
                         int n, int direction) {
    complexDoubleDevice* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexDoubleDevice));
    complexDoubleDevice* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexDoubleDevice));
    gpuMemcpy(d_buff2, Buff2, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyHostToDevice);
    gpu_reorder(d_buff1, d_buff2, n, direction);
    gpuMemcpy(Buff1, d_buff1, m_nlocal * sizeof(complexDoubleHost),
              cudaMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void GPUReorder::reorder(complexFloatHost* Buff1, complexFloatHost* Buff2,
                         int n, int direction) {
    complexFloatDevice* d_buff1;
    swfftAlloc(&d_buff1, m_nlocal * sizeof(complexFloatDevice));
    complexFloatDevice* d_buff2;
    swfftAlloc(&d_buff2, m_nlocal * sizeof(complexFloatDevice));
    gpuMemcpy(d_buff2, Buff2, m_nlocal * sizeof(complexFloatHost),
              cudaMemcpyHostToDevice);
    gpu_reorder(d_buff1, d_buff2, n, direction);
    gpuMemcpy(Buff1, d_buff1, m_nlocal * sizeof(complexFloatHost),
              cudaMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

} // namespace A2A
} // namespace SWFFT

#endif
#endif