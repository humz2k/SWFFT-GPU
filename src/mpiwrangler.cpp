#include "mpiwrangler.hpp"
#include <stdlib.h>

template<class T>
void base_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    MPI_Alltoall(buff1,n*sizeof(T),MPI_BYTE,buff2,n*sizeof(T),MPI_BYTE,comm);
}

template<class T>
void base_irecv(T* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req){
    MPI_Irecv(buff,count * sizeof(T),MPI_BYTE,source,tag,comm,req);
}

#ifdef SWFFT_GPU
template<class T>
void CPUMPI::gpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    int world_size; MPI_Comm_size(comm,&world_size);
    int sz = world_size * n * sizeof(T);
    T* h_buff1 = (T*)get_h_buff1(sz);
    T* h_buff2 = (T*)get_h_buff2(sz);
    gpuMemcpy(h_buff1,buff1,sz,gpuMemcpyDeviceToHost);
    base_alltoall(h_buff1,h_buff2,n,comm);
    gpuMemcpy(buff2,h_buff2,sz,gpuMemcpyHostToDevice);
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

CPUMPI::CPUMPI() : last_size(0){

}

CPUMPI::~CPUMPI(){
    if (last_size != 0){
        free(_h_buff1);
        free(_h_buff2);
    }
}

void CPUMPI::alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm){
    base_alltoall(buff1,buff2,n,comm);
}

void CPUMPI::alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm){
    base_alltoall(buff1,buff2,n,comm);
}

#ifdef SWFFT_GPU
inline void* CPUMPI::get_h_buff1(size_t sz){
    if (last_size == 0){
        _h_buff1 = malloc(sz);
        _h_buff2 = malloc(sz);
        last_size = sz;
        return _h_buff1;
    }
    if (last_size < sz){
        free(_h_buff1);
        free(_h_buff2);
        _h_buff1 = malloc(sz);
        _h_buff2 = malloc(sz);
        last_size = sz;
        return _h_buff1;
    }
    return _h_buff1;
}

inline void* CPUMPI::get_h_buff2(size_t sz){
    if (last_size == 0){
        _h_buff1 = malloc(sz);
        _h_buff2 = malloc(sz);
        last_size = sz;
        return _h_buff2;
    }
    if (last_size < sz){
        free(_h_buff1);
        free(_h_buff2);
        _h_buff1 = malloc(sz);
        _h_buff2 = malloc(sz);
        last_size = sz;
        return _h_buff2;
    }
    return _h_buff2;
}

void CPUMPI::alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm){
    gpu_memcpy_alltoall(buff1,buff2,n,comm);

}

void CPUMPI::alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm){
    gpu_memcpy_alltoall(buff1,buff2,n,comm);

}
#endif

#ifdef SWFFT_GPU
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