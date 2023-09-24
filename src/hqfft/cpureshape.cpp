#ifdef SWFFT_HQFFT
#include "hqfft_reorder.hpp"

namespace SWFFT{
namespace HQFFT{

CPUReshape::CPUReshape(){}
CPUReshape::~CPUReshape(){}

template<class T>
inline void CPUReshape::_reshape(const T* __restrict buff1, T* __restrict buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;

        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[new_idx] = buff1[i];
    }
}

template<class T>
inline void CPUReshape::_inverse_reshape(const T* __restrict buff1, T* __restrict buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++){
        int mini_pencil_id = i / mini_pencil_size;

        int rank = i / send_per_rank;

        int rank_offset = rank * mini_pencil_size;

        int pencil_offset = (mini_pencil_id % pencils_per_rank) * mini_pencil_size * n_recvs;

        int local_offset = i % mini_pencil_size;

        int new_idx = rank_offset + pencil_offset + local_offset;

        buff2[i] = buff1[new_idx];
    }
}

void CPUReshape::reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    _reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

void CPUReshape::inverse_reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    _inverse_reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

void CPUReshape::reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    _reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

void CPUReshape::inverse_reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    _inverse_reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

#ifdef SWFFT_GPU
void CPUReshape::reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexDoubleHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _reshape(h_buff1,h_buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::inverse_reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexDoubleHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _inverse_reshape(h_buff1,h_buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexFloatHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _reshape(h_buff1,h_buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::inverse_reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexFloatHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _inverse_reshape(h_buff1,h_buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}
#endif

template<class T>
inline void CPUReshape::_unreshape(const T* __restrict buff1, T* __restrict buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[new_idx] = buff1[i];
    }
}

template<class T>
inline void CPUReshape::_inverse_unreshape(const T* __restrict buff1, T* __restrict buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++){
        int x = i / (y_dim * z_dim);
        int y = (i - (x * y_dim * z_dim)) / z_dim;
        int z = (i - (x * y_dim * z_dim)) - y * z_dim;
        int new_idx = z * x_dim * y_dim + x * y_dim + y;

        buff2[i] = buff1[new_idx];
    }
}

void CPUReshape::unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    _unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

void CPUReshape::inverse_unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    _inverse_unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

void CPUReshape::unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    _unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

void CPUReshape::inverse_unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    _inverse_unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

#ifdef SWFFT_GPU
void CPUReshape::unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexDoubleHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _unreshape(h_buff1,h_buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::inverse_unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexDoubleHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _inverse_unreshape(h_buff1,h_buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexFloatHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _unreshape(h_buff1,h_buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::inverse_unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexFloatHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _inverse_unreshape(h_buff1,h_buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}
#endif

template<class T>
inline void CPUReshape::_reshape_final(const T* __restrict buff1, T* __restrict buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    #pragma omp parallel for
    for (int i = 0; i < nlocal; i++){
        int3 local_dims = make_int3(local_grid_size[0],local_grid_size[1] / ny,local_grid_size[2] / nz); //per rank dims

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

        int z_offset = (local_grid_size[2] / nz) * z_coord;

        int y_offset = (local_grid_size[1] / ny) * y_coord;

        int3 global_coords = make_int3(local_coords.x,local_coords.y + y_offset,local_coords.z + z_offset);

        int new_idx = global_coords.x * local_grid_size[1] * local_grid_size[2] + global_coords.y * local_grid_size[2] + global_coords.z;

        buff2[new_idx] = buff1[i];
    }
}

void CPUReshape::reshape_final(complexDoubleHost* buff1, complexDoubleHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    _reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
}

void CPUReshape::reshape_final(complexFloatHost* buff1, complexFloatHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    _reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
}

#ifdef SWFFT_GPU
void CPUReshape::reshape_final(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    size_t sz = sizeof(complexDoubleDevice) * nlocal;
    complexDoubleHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexDoubleHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void CPUReshape::reshape_final(complexFloatDevice* buff1, complexFloatDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize){
    size_t sz = sizeof(complexFloatDevice) * nlocal;
    complexFloatHost* h_buff1; swfftAlloc(&h_buff1,sz);
    complexFloatHost* h_buff2; swfftAlloc(&h_buff2,sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    _reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}
#endif

}
}

#endif