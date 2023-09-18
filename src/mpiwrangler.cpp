#include "mpiwrangler.hpp"
#include <stdlib.h>

template<class T>
void base_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    MPI_Alltoall(buff1,n*sizeof(T),MPI_BYTE,buff2,n*sizeof(T),MPI_BYTE,comm);
}

#ifdef GPU
template<class T>
void gpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    int world_size; MPI_Comm_size(comm,&world_size);
    int sz = world_size * n * sizeof(T);
    T* h_buff1 = (T*)malloc(sz);
    T* h_buff2 = (T*)malloc(sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    base_alltoall(h_buff1,h_buff2,n,comm);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
    free(h_buff1);
    free(h_buff2);
}

#ifndef nocudampi
template<class T>
void cpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    int world_size; MPI_Comm_size(comm,&world_size);
    int sz = world_size * n * sizeof(T);
    T* d_buff1; gpuMalloc(&h_buff1,sz);
    T* d_buff2; gpuMalloc(&h_buff2,sz);
    gpuMemcpy(d_buff1,buff1,sz,gpuMemcpyHostToDevice);
    base_alltoall(d_buff1,d_buff2,n,comm);
    gpuMemcpy(buff2,d_buff2,sz,gpuMemcpyDeviceToHost);
    gpuFree(d_buff1);
    gpuFree(d_buff2);
}
#endif
#endif

CPUMPI::CPUMPI(){

}

CPUMPI::~CPUMPI(){

}

void CPUMPI::alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm){
    base_alltoall(buff1,buff2,n,comm);
}

void CPUMPI::alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm){
    base_alltoall(buff1,buff2,n,comm);
}

#ifdef GPU
void CPUMPI::alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm){
    gpu_memcpy_alltoall(buff1,buff2,n,comm);
}

void CPUMPI::alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm){
    gpu_memcpy_alltoall(buff1,buff2,n,comm);
}
#endif

#ifdef GPU
#ifndef nocudampi

GPUMPI::GPUMPI(){

}

GPUMPI::~GPUMPI(){

}

void GPUMPI::alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm){
    base_alltoall(buff1,buff2,n,comm);
}

void GPUMPI::alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm){
    base_alltoall(buff1,buff2,n,comm);
}

void GPUMPI::alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm){
    cpu_memcpy_alltoall(buff1,buff2,n,comm);
}

void GPUMPI::alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm){
    cpu_memcpy_alltoall(buff1,buff2,n,comm);
}

#endif
#endif