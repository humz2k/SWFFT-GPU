#ifdef ALLTOALL
#include "a2atopencils.hpp"
namespace A2A{
__forceinline__ __device__ int calc_mini_pencil_idx(int i, int mini_pencils_per_rank, int world_size, int mini_pencil_size){
    int global_mini_pencil_id = i / mini_pencil_size;
    int rank = global_mini_pencil_id / mini_pencils_per_rank;
    int local_mini_pencil_id = global_mini_pencil_id % mini_pencils_per_rank;
    int global_mini_pencil_offset = world_size * mini_pencil_size;
    int my_mini_pencil_offset = local_mini_pencil_id * global_mini_pencil_offset;
    int my_pencil_start = rank * mini_pencil_size + my_mini_pencil_offset;
    int sub_mini_pencil_idx = i % mini_pencil_size;
    int new_idx = my_pencil_start + sub_mini_pencil_idx;
    return new_idx;
}

template<class T>
__global__ void reorder_forwards_kernel(const T* __restrict src, T* __restrict dest, int mini_pencils_per_rank, int world_size, int mini_pencil_size, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)return;

    int new_idx = calc_mini_pencil_idx(i,mini_pencils_per_rank,world_size,mini_pencil_size);
    dest[new_idx] = __ldg(&src[i]);

}

template<class T>
__global__ void reorder_backwards_kernel(const T* __restrict src, T* __restrict dest, int mini_pencils_per_rank, int world_size, int mini_pencil_size, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)return;
    int new_idx = calc_mini_pencil_idx(i,mini_pencils_per_rank,world_size,mini_pencil_size);
    //dest[i] = __ldg(&src[calc_mini_pencil_idx(i,mini_pencils_per_rank,world_size,mini_pencil_size)]);
    dest[i] = __ldg(&src[new_idx]);

}

template<class T>
void launch_reorder_forward(T* src, T* dest, int mini_pencils_per_rank, int world_size, int mini_pencil_size, int blockSize, gpuStream_t stream){
    int n = mini_pencils_per_rank * world_size * mini_pencil_size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    reorder_forwards_kernel<<<numBlocks,blockSize,0,stream>>>(src,dest,mini_pencils_per_rank,world_size,mini_pencil_size,n);

    //cudaDeviceSynchronize();
    //cudaStreamSynchronize(stream);
    
}
template<class T>
void launch_reorder_backward(T* src, T* dest, int mini_pencils_per_rank, int world_size, int mini_pencil_size, int blockSize, gpuStream_t stream){
    int n = mini_pencils_per_rank * world_size * mini_pencil_size;
    int numBlocks = (n + (blockSize - 1))/blockSize;

    reorder_backwards_kernel<<<numBlocks,blockSize,0,stream>>>(src,dest,mini_pencils_per_rank,world_size,mini_pencil_size,n);

    //cudaDeviceSynchronize();
    //cudaStreamSynchronize(stream);
    
}

template void launch_reorder_forward<complexDoubleDevice>(complexDoubleDevice*, complexDoubleDevice*, int,int,int,int,gpuStream_t);
template void launch_reorder_forward<complexFloatDevice>(complexFloatDevice*, complexFloatDevice*, int,int,int,int,gpuStream_t);

template void launch_reorder_backward<complexDoubleDevice>(complexDoubleDevice*, complexDoubleDevice*, int,int,int,int,gpuStream_t);
template void launch_reorder_backward<complexFloatDevice>(complexFloatDevice*, complexFloatDevice*, int,int,int,int,gpuStream_t);

}
#endif