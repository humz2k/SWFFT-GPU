#ifdef SWFFT_HQFFT
#ifdef SWFFT_GPU
#include "hqfft_reorder.hpp"

namespace SWFFT{
namespace HQFFT{

template<class T>
__global__ void reshape_kernel(const T* __restrict buff1, T* __restrict buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int mini_pencil_id = i / mini_pencil_size;

    int rank = i / send_per_rank;

    int rank_offset = rank * mini_pencil_size;

    int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

    int local_offset = i % mini_pencil_size;

    int new_idx = rank_offset + pencil_offset + local_offset;

    buff2[new_idx] = __ldg(&buff1[i]);
}

template<class T>
inline void GPUReshape::_reshape(T* buff1, T* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;
    gpuLaunch(reshape_kernel,numBlocks,blockSize,buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal);
}

void GPUReshape::reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    _reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

void GPUReshape::reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    _reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

void GPUReshape::reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleDevice* d_buff1; swfftAlloc(&d_buff1,sz);
    complexDoubleDevice* d_buff2; swfftAlloc(&d_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    _reshape(d_buff1,d_buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void GPUReshape::reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatDevice* d_buff1; swfftAlloc(&d_buff1,sz);
    complexFloatDevice* d_buff2; swfftAlloc(&d_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    _reshape(d_buff1,d_buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

template<class T>
__global__ void unreshape_kernel(const T* __restrict buff1, T* __restrict buff2, int z_dim, int x_dim, int y_dim, int nlocal){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int x = i / (y_dim * z_dim);
    int y = (i - (x * y_dim * z_dim)) / z_dim;
    int z = (i - (x * y_dim * z_dim)) - y * z_dim;
    int new_idx = z * x_dim * y_dim + x * y_dim + y;

    buff2[new_idx] = __ldg(&buff1[i]);
}

template<class T>
inline void GPUReshape::_unreshape(T* buff1, T* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;
    gpuLaunch(unreshape_kernel,numBlocks,blockSize,buff1,buff2,z_dim,x_dim,y_dim,nlocal);
}

void GPUReshape::unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    _unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

void GPUReshape::unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    _unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

void GPUReshape::unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleDevice* d_buff1; swfftAlloc(&d_buff1,sz);
    complexDoubleDevice* d_buff2; swfftAlloc(&d_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    _unreshape(d_buff1,d_buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void GPUReshape::unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatDevice* d_buff1; swfftAlloc(&d_buff1,sz);
    complexFloatDevice* d_buff2; swfftAlloc(&d_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    _unreshape(d_buff1,d_buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

template<class T>
__global__ void reshape_final_kernel(const T* __restrict buff1, T* __restrict buff2, int ny, int nz, int3 local_grid_size, int nlocal){
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int3 local_dims = make_int3(local_grid_size.x,local_grid_size.y / ny,local_grid_size.z / nz); //per rank dims

    int n_recvs = ny * nz; //where we recieve from in each direction.
    int per_rank = nlocal / n_recvs; //how many per rank we have recieved
    int rank = i / per_rank; //which rank I am from

    int i_local = i % per_rank; //my idx local to the rank I am from

    int3 local_coords;

    local_coords.x = i_local / (local_dims.y * local_dims.z);
    local_coords.y = (i_local - local_coords.x * local_dims.y * local_dims.z) / local_dims.z;
    local_coords.z = (i_local - local_coords.x * local_dims.y * local_dims.z) - local_coords.y * local_dims.z;

    int z_coord = rank / ny; //z is slow index for sends

    int y_coord = rank - z_coord * ny; //y is fast index for sends

    int z_offset = (local_grid_size.z / nz) * z_coord;

    int y_offset = (local_grid_size.y / ny) * y_coord;

    int3 global_coords = make_int3(local_coords.x,local_coords.y + y_offset,local_coords.z + z_offset);

    int new_idx = global_coords.x * local_grid_size.y * local_grid_size.z + global_coords.y * local_grid_size.z + global_coords.z;

    buff2[new_idx] = __ldg(&buff1[i]);
    
}

template<class T>
inline void GPUReshape::_reshape_final(T* buff1, T* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    int numBlocks = (nlocal + (blockSize - 1))/blockSize;
    int3 local_grid_size_vec = make_int3(local_grid_size[0],local_grid_size[1],local_grid_size[2]);
    gpuLaunch(reshape_final_kernel,numBlocks,blockSize,buff1,buff2,ny,nz,local_grid_size_vec,nlocal);
}

void GPUReshape::reshape_final(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    _reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
}

void GPUReshape::reshape_final(complexFloatDevice* buff1, complexFloatDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    _reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
}

void GPUReshape::reshape_final(complexDoubleHost* buff1, complexDoubleHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleDevice* d_buff1; swfftAlloc(&d_buff1,sz);
    complexDoubleDevice* d_buff2; swfftAlloc(&d_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    _reshape_final(d_buff1,d_buff2,ny,nz,local_grid_size,nlocal,blockSize);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

void GPUReshape::reshape_final(complexFloatHost* buff1, complexFloatHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatDevice* d_buff1; swfftAlloc(&d_buff1,sz);
    complexFloatDevice* d_buff2; swfftAlloc(&d_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    _reshape_final(d_buff1,d_buff2,ny,nz,local_grid_size,nlocal,blockSize);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    swfftFree(d_buff1);
    swfftFree(d_buff2);
}

}
}
#endif
#endif