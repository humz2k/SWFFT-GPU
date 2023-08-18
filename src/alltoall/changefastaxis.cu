#ifdef ALLTOALL
#include "changefastaxis.hpp"
namespace A2A{

template<class T>
__global__ void d_fast_z_to_x(const T* __restrict source, T* __restrict dest, int lgridx, int lgridy, int lgridz, int nlocal){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < nlocal){

        int i = idx / (lgridx * lgridy);
        int j = (idx - (i * lgridx * lgridy)) / lgridx;
        int k = idx - (i * (lgridx * lgridy)) - (j * lgridx);

        int dest_index = i*lgridy*lgridx + j*lgridx + k;
        int source_index = k*lgridy*lgridz + j*lgridz + i;

        dest[dest_index] = source[source_index];
        //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

    }

}

template<class T>
void launch_fast_z_to_x(T* source, T* dest, int* local_grid_size, int blockSize, int numBlocks, int nlocal){

    d_fast_z_to_x<<<numBlocks,blockSize,0,0>>>(source, dest, local_grid_size[0], local_grid_size[1], local_grid_size[2], nlocal);

    gpuDeviceSynchronize();

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

        dest[dest_index] = source[source_index];
        //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

    }

}

template<class T>
void launch_fast_x_to_y(T* source, T* dest, int* local_grid_size, int blockSize, int numBlocks, int nlocal){

    d_fast_x_to_y<<<numBlocks,blockSize,0,0>>>(source, dest, local_grid_size[0], local_grid_size[1], local_grid_size[2], nlocal);

    gpuDeviceSynchronize();

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

        dest[dest_index] = source[source_index];
        //dest[dest_index * 2 + 1] = source[source_index * 2 + 1];

    }

}

template<class T>
void launch_fast_y_to_z(T* source, T* dest, int* local_grid_size, int blockSize, int numBlocks, int nlocal){

    d_fast_y_to_z<<<numBlocks,blockSize,0,0>>>(source, dest, local_grid_size[0], local_grid_size[1], local_grid_size[2], nlocal);

    gpuDeviceSynchronize();

}

template void launch_fast_y_to_z<complexDouble>(complexDouble*, complexDouble*, int*,int,int,int);
template void launch_fast_y_to_z<complexFloat>(complexFloat*, complexFloat*, int*,int,int,int);
template void launch_fast_x_to_y<complexDouble>(complexDouble*, complexDouble*, int*,int,int,int);
template void launch_fast_x_to_y<complexFloat>(complexFloat*, complexFloat*, int*,int,int,int);
template void launch_fast_z_to_x<complexDouble>(complexDouble*, complexDouble*, int*,int,int,int);
template void launch_fast_z_to_x<complexFloat>(complexFloat*, complexFloat*, int*,int,int,int);

}
#endif