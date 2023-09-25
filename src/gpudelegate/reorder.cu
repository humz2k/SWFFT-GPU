#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
#ifdef SWFFT_GPUDELEGATE

#include "gpudelegate_reorder.hpp"

namespace SWFFT{
namespace GPUDELEGATE{

template<class T>
__global__ void reorder_forwards_kernel(const T* __restrict buff1, T* __restrict buff2, int3 grid_start, int3 local_grid_size, int3 ng, int nlocal){

    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int3 my_idx;
    my_idx.x = i / (local_grid_size.y * local_grid_size.z);
    my_idx.y = (i - (my_idx.x * local_grid_size.y * local_grid_size.z)) / local_grid_size.z;
    my_idx.z = (i - (my_idx.x * local_grid_size.y * local_grid_size.z)) - my_idx.y * local_grid_size.z;

    int3 new_idx;
    new_idx.x = my_idx.x + grid_start.x;
    new_idx.y = my_idx.y + grid_start.y;
    new_idx.z = my_idx.z + grid_start.z;

    int j = new_idx.x * ng.y * ng.z + new_idx.y * ng.z + new_idx.z;

    buff2[j] = __ldg(&buff1[i]);

}

template<class T>
__global__ void reorder_backwards_kernel(const T* __restrict buff1, T* __restrict buff2, int3 grid_start, int3 local_grid_size, int3 ng, int nlocal){

    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= nlocal)return;

    int3 my_idx;
    my_idx.x = i / (local_grid_size.y * local_grid_size.z);
    my_idx.y = (i - (my_idx.x * local_grid_size.y * local_grid_size.z)) / local_grid_size.z;
    my_idx.z = (i - (my_idx.x * local_grid_size.y * local_grid_size.z)) - my_idx.y * local_grid_size.z;

    int3 new_idx;
    new_idx.x = my_idx.x + grid_start.x;
    new_idx.y = my_idx.y + grid_start.y;
    new_idx.z = my_idx.z + grid_start.z;

    int j = new_idx.x * ng.y * ng.z + new_idx.y * ng.z + new_idx.z;

    buff2[i] = __ldg(&buff1[j]);

}

template<class T>
inline void _reorder(int direction, T* buff1, T* buff2, int3 ng, int3 local_grid_size, int3 dims, int rank, gpuStream_t stream, int blockSize){

    int3 coords;
    coords.x = rank / (dims.y * dims.z);
    coords.y = (rank - (coords.x * dims.y * dims.z)) / dims.z;
    coords.z = (rank - (coords.x * dims.y * dims.z)) - coords.y * dims.z;

    int3 grid_start;
    grid_start.x = coords.x * local_grid_size.x;
    grid_start.y = coords.y * local_grid_size.y;
    grid_start.z = coords.z * local_grid_size.z;

    int nlocal = local_grid_size.x * local_grid_size.y * local_grid_size.z;

    int numBlocks = (nlocal + (blockSize - 1))/blockSize;

    if(direction == 0){
        reorder_forwards_kernel<<<numBlocks,blockSize,0,stream>>>(buff1,buff2,grid_start,local_grid_size,ng,nlocal);
    } else {
        reorder_backwards_kernel<<<numBlocks,blockSize,0,stream>>>(buff1,buff2,grid_start,local_grid_size,ng,nlocal);
    }

}

void reorder(int direction, complexDoubleDevice* buff1, complexDoubleDevice* buff2, int3 ng, int3 local_grid_size, int3 dims, int rank, gpuStream_t stream, int blockSize){

    _reorder(direction,buff1,buff2,ng,local_grid_size,dims,rank,stream,blockSize);
    
}

void reorder(int direction, complexFloatDevice* buff1, complexFloatDevice* buff2, int3 ng, int3 local_grid_size, int3 dims, int rank, gpuStream_t stream, int blockSize){

    _reorder(direction,buff1,buff2,ng,local_grid_size,dims,rank,stream,blockSize);
    
}



}
}

#endif
#endif
#endif