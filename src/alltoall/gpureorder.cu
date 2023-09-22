#ifdef SWFFT_ALLTOALL
#ifdef SWFFT_GPU
#include "alltoall_reorder.hpp"

namespace A2A{
    namespace GPUREORDER{
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
            dest[i] = __ldg(&src[new_idx]);

        }

        template<class T>
        __global__ void d_fast_z_to_x(const T* __restrict source, T* __restrict dest, int lgridx, int lgridy, int lgridz, int nlocal){

            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            if (idx < nlocal){

                int i = idx / (lgridx * lgridy);
                int j = (idx - (i * lgridx * lgridy)) / lgridx;
                int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

                int dest_index = i*lgridy*lgridx + j*lgridx + k;
                int source_index = k*lgridy*lgridz + j*lgridz + i;

                dest[dest_index] = __ldg(&source[source_index]);
                //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

            }

        }

        template<class T>
        __global__ void d_fast_x_to_z(const T* __restrict source, T* __restrict dest, int lgridx, int lgridy, int lgridz, int nlocal){

            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            if (idx < nlocal){

                int i = idx / (lgridx * lgridy);
                int j = (idx - (i * lgridx * lgridy)) / lgridx;
                int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

                int dest_index = i*lgridy*lgridx + j*lgridx + k;
                int source_index = k*lgridy*lgridz + j*lgridz + i;

                dest[source_index] = __ldg(&source[dest_index]);
                //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

            }

        }


        template<class T>
        __global__ void d_fast_x_to_y(const T* __restrict source, T* __restrict dest, int lgridx, int lgridy, int lgridz, int nlocal){
            
            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            if (idx < nlocal){

                int i = idx / (lgridz * lgridy);
                int j = (idx - (i * lgridz * lgridy)) / lgridy;
                int k = idx - (i * (lgridz * lgridy)) - (j * lgridy);

                int dest_index = i*lgridz*lgridy + j*lgridy + k;
                int source_index = j*lgridx*lgridy + k*lgridx + i;

                dest[dest_index] = __ldg(&source[source_index]);
                //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

            }

        }

        template<class T>
        __global__ void d_fast_y_to_z(const T* __restrict source, T* __restrict dest, int lgridx, int lgridy, int lgridz, int nlocal){

            int idx = blockDim.x * blockIdx.x + threadIdx.x;

            if (idx < nlocal){

                int i = idx / (lgridz * lgridy);
                int j = (idx - (i * lgridz * lgridy)) / lgridz;
                int k = idx - (i * (lgridz * lgridy)) - (j * lgridz);

                int dest_index = i*lgridz*lgridy + j*lgridz + k;
                int source_index = i*lgridz*lgridy + k*lgridy + j;

                dest[dest_index] = __ldg(&source[source_index]);
                //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

            }

        }
    }

    GPUReorder::GPUReorder(int3 ng_, int3 dims_, int3 coords_, int blockSize_) : ng(ng_), dims(dims_), coords(coords_), blockSize(blockSize_){
        local_grid.x = ng.x / dims.x;
        local_grid.y = ng.y / dims.y;
        local_grid.z = ng.z / dims.z;
        nlocal = local_grid.x * local_grid.y * local_grid.z;
        local_grid_size[0] = local_grid.x;
        local_grid_size[1] = local_grid.y;
        local_grid_size[2] = local_grid.z;
        world_size = dims.x * dims.y * dims.z;
    }
    GPUReorder::~GPUReorder(){};
    GPUReorder::GPUReorder(){};

    template<class T>
    inline void GPUReorder::gpu_shuffle_indices(T* Buff1, T* Buff2, int n){
        int numBlocks = (nlocal + (blockSize - 1)) / blockSize;
        switch(n){
            case 0:
                gpuLaunch(GPUREORDER::d_fast_z_to_x,numBlocks,blockSize,Buff2,Buff1,local_grid.x,local_grid.y,local_grid.z,nlocal);
                //GPUREORDER::d_fast_z_to_x<<<numBlocks,blockSize>>>();
                break;
            case 1:
                gpuLaunch(GPUREORDER::d_fast_x_to_y,numBlocks,blockSize,Buff2,Buff1,local_grid.x,local_grid.y,local_grid.z,nlocal);
                //GPUREORDER::d_fast_x_to_y<<<numBlocks,blockSize>>>(Buff2,Buff1,local_grid.z,local_grid.y,local_grid.z,nlocal);
                break;
            case 2:
                gpuLaunch(GPUREORDER::d_fast_y_to_z,numBlocks,blockSize,Buff2,Buff1,local_grid.x,local_grid.y,local_grid.z,nlocal);
                //GPUREORDER::d_fast_y_to_z<<<numBlocks,blockSize>>>(Buff2,Buff1,local_grid.z,local_grid.y,local_grid.z,nlocal);
                break;
            case 3:
                gpuLaunch(GPUREORDER::d_fast_x_to_z,numBlocks,blockSize,Buff2,Buff1,local_grid.x,local_grid.y,local_grid.z,nlocal);
                //GPUREORDER::d_fast_z_to_x<<<numBlocks,blockSize>>>();
                break;
        }
    }

    void GPUReorder::shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n){
        gpu_shuffle_indices(Buff1,Buff2,n);
    }

    void GPUReorder::shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n){
        gpu_shuffle_indices(Buff1,Buff2,n);
    }

    void GPUReorder::shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n){
        complexDoubleDevice* d_buff1; swfftAlloc(&d_buff1,nlocal*sizeof(complexDoubleDevice));
        complexDoubleDevice* d_buff2; swfftAlloc(&d_buff2,nlocal*sizeof(complexDoubleDevice));
        gpuMemcpy(d_buff2,Buff2,nlocal*sizeof(complexDoubleHost),cudaMemcpyHostToDevice);
        gpu_shuffle_indices(d_buff1,d_buff2,n);
        gpuMemcpy(Buff1,d_buff1,nlocal*sizeof(complexDoubleHost),cudaMemcpyDeviceToHost);
        swfftFree(d_buff1);
        swfftFree(d_buff2);
    }

    void GPUReorder::shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n){
        complexFloatDevice* d_buff1; swfftAlloc(&d_buff1,nlocal*sizeof(complexFloatHost));
        complexFloatDevice* d_buff2; swfftAlloc(&d_buff2,nlocal*sizeof(complexFloatHost));
        gpuMemcpy(d_buff2,Buff2,nlocal*sizeof(complexFloatDevice),cudaMemcpyHostToDevice);
        gpu_shuffle_indices(d_buff1,d_buff2,n);
        gpuMemcpy(Buff1,d_buff1,nlocal*sizeof(complexFloatDevice),cudaMemcpyDeviceToHost);
        swfftFree(d_buff1);
        swfftFree(d_buff2);
    }

    template<class T>
    inline void GPUReorder::gpu_reorder(T* Buff1, T* Buff2, int n, int direction){
        int dim = (n+2)%3;
        int nsends = (nlocal / world_size);
        int numBlocks = (nlocal + (blockSize - 1)) / blockSize;
        if (direction == 0){
            gpuLaunch(GPUREORDER::reorder_forwards_kernel,numBlocks,blockSize,Buff2,Buff1,nsends / local_grid_size[dim],world_size,local_grid_size[dim],nlocal);
            //GPUREORDER::reorder_forwards_kernel<<<numBlocks,blockSize>>>(Buff2,Buff1,nsends / local_grid_size[dim],world_size,local_grid_size[dim],nlocal);
        } else {
            gpuLaunch(GPUREORDER::reorder_backwards_kernel,numBlocks,blockSize,Buff2,Buff1,nsends / local_grid_size[dim],world_size,local_grid_size[dim],nlocal);
            //GPUREORDER::reorder_backwards_kernel<<<numBlocks,blockSize>>>(Buff2,Buff1,nsends / local_grid_size[dim],world_size,local_grid_size[dim],nlocal);
        }
    }

    void GPUReorder::reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n, int direction){
        gpu_reorder(Buff1,Buff2,n,direction);
    }

    void GPUReorder::reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n, int direction){
        gpu_reorder(Buff1,Buff2,n,direction);
    }

    void GPUReorder::reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n, int direction){
        complexDoubleDevice* d_buff1; swfftAlloc(&d_buff1,nlocal*sizeof(complexDoubleDevice));
        complexDoubleDevice* d_buff2; swfftAlloc(&d_buff2,nlocal*sizeof(complexDoubleDevice));
        gpuMemcpy(d_buff2,Buff2,nlocal*sizeof(complexDoubleHost),cudaMemcpyHostToDevice);
        gpu_reorder(d_buff1,d_buff2,n,direction);
        gpuMemcpy(Buff1,d_buff1,nlocal*sizeof(complexDoubleHost),cudaMemcpyDeviceToHost);
        swfftFree(d_buff1);
        swfftFree(d_buff2);
    }

    void GPUReorder::reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n, int direction){
        complexFloatDevice* d_buff1; swfftAlloc(&d_buff1,nlocal*sizeof(complexFloatDevice));
        complexFloatDevice* d_buff2; swfftAlloc(&d_buff2,nlocal*sizeof(complexFloatDevice));
        gpuMemcpy(d_buff2,Buff2,nlocal*sizeof(complexFloatHost),cudaMemcpyHostToDevice);
        gpu_reorder(d_buff1,d_buff2,n,direction);
        gpuMemcpy(Buff1,d_buff1,nlocal*sizeof(complexFloatHost),cudaMemcpyDeviceToHost);
        swfftFree(d_buff1);
        swfftFree(d_buff2);
    }

}

#endif
#endif