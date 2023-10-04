#include "mpiwrangler.hpp"
#include <stdlib.h>
#include <stdio.h>
namespace SWFFT{

void CPUMPI::query(){
    printf("Using CPUMPI\n");
}

template<class T>
inline void base_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    MPI_Alltoall(buff1,n*sizeof(T),MPI_BYTE,buff2,n*sizeof(T),MPI_BYTE,comm);
}

template<class T>
inline void base_sendrecv(T* send_buff, int sendcount, int dest, int sendtag, T* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    MPI_Sendrecv(send_buff,sendcount * sizeof(T), MPI_BYTE, dest, sendtag, recv_buff, recvcount * sizeof(T), MPI_BYTE, source, recvtag, comm, MPI_STATUS_IGNORE);
}

/*template<class T>
void base_irecv(T* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req){
    MPI_Irecv(buff,count * sizeof(T),MPI_BYTE,source,tag,comm,req);
}*/

void CPUMPI::sendrecv(complexDoubleHost* send_buff, int sendcount, int dest, int sendtag, complexDoubleHost* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    base_sendrecv(send_buff,sendcount,dest,sendtag,recv_buff,recvcount,source,recvtag,comm);
}

void CPUMPI::sendrecv(complexFloatHost* send_buff, int sendcount, int dest, int sendtag, complexFloatHost* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    base_sendrecv(send_buff,sendcount,dest,sendtag,recv_buff,recvcount,source,recvtag,comm);
}

#ifdef SWFFT_GPU
void CPUMPI::sendrecv(complexDoubleDevice* send_buff, int sendcount, int dest, int sendtag, complexDoubleDevice* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    size_t send_size = sendcount * sizeof(complexDoubleDevice);
    size_t recv_size = recvcount * sizeof(complexDoubleDevice);
    complexDoubleHost* h_buff1 = (complexDoubleHost*)get_h_buff1(send_size);
    complexDoubleHost* h_buff2 = (complexDoubleHost*)get_h_buff2(recv_size);
    gpuMemcpy(h_buff1,send_buff,send_size,gpuMemcpyDeviceToHost);
    base_sendrecv(h_buff1,sendcount,dest,sendtag,h_buff2,recvcount,source,recvtag,comm);
    gpuMemcpy(recv_buff,h_buff2,recv_size,gpuMemcpyHostToDevice);
}

void CPUMPI::sendrecv(complexFloatDevice* send_buff, int sendcount, int dest, int sendtag, complexFloatDevice* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    size_t send_size = sendcount * sizeof(complexFloatDevice);
    size_t recv_size = recvcount * sizeof(complexFloatDevice);
    complexFloatHost* h_buff1 = (complexFloatHost*)get_h_buff1(send_size);
    complexFloatHost* h_buff2 = (complexFloatHost*)get_h_buff2(recv_size);
    gpuMemcpy(h_buff1,send_buff,send_size,gpuMemcpyDeviceToHost);
    base_sendrecv(h_buff1,sendcount,dest,sendtag,h_buff2,recvcount,source,recvtag,comm);
    gpuMemcpy(recv_buff,h_buff2,recv_size,gpuMemcpyHostToDevice);
}
#endif

CPUIsend<complexDoubleHost>* CPUMPI::isend(complexDoubleHost* buff, int n, int dest, int tag, MPI_Comm comm){
    CPUIsend<complexDoubleHost>* out = new CPUIsend<complexDoubleHost>(buff,n,dest,tag,comm);
    return out;
}

CPUIsend<complexFloatHost>* CPUMPI::isend(complexFloatHost* buff, int n, int dest, int tag, MPI_Comm comm){
    CPUIsend<complexFloatHost>* out = new CPUIsend<complexFloatHost>(buff,n,dest,tag,comm);
    return out;
}

CPUIrecv<complexDoubleHost>* CPUMPI::irecv(complexDoubleHost* buff, int n, int source, int tag, MPI_Comm comm){
    CPUIrecv<complexDoubleHost>* out = new CPUIrecv<complexDoubleHost>(buff,n,source,tag,comm);
    return out;
}

CPUIrecv<complexFloatHost>* CPUMPI::irecv(complexFloatHost* buff, int n, int source, int tag, MPI_Comm comm){
    CPUIrecv<complexFloatHost>* out = new CPUIrecv<complexFloatHost>(buff,n,source,tag,comm);
    return out;
}

#ifdef SWFFT_GPU
CPUIsend<complexDoubleDevice>* CPUMPI::isend(complexDoubleDevice* buff, int n, int dest, int tag, MPI_Comm comm){
    //printf("make isend!\n");
    CPUIsend<complexDoubleDevice>* out = new CPUIsend<complexDoubleDevice>(buff,n,dest,tag,comm);
    return out;
}

CPUIsend<complexFloatDevice>* CPUMPI::isend(complexFloatDevice* buff, int n, int dest, int tag, MPI_Comm comm){
    CPUIsend<complexFloatDevice>* out = new CPUIsend<complexFloatDevice>(buff,n,dest,tag,comm);
    return out;
}

CPUIrecv<complexDoubleDevice>* CPUMPI::irecv(complexDoubleDevice* buff, int n, int source, int tag, MPI_Comm comm){
    //printf("make irecv\n");
    CPUIrecv<complexDoubleDevice>* out = new CPUIrecv<complexDoubleDevice>(buff,n,source,tag,comm);//(buff,n,source,tag,comm);
    return out;
}

CPUIrecv<complexFloatDevice>* CPUMPI::irecv(complexFloatDevice* buff, int n, int source, int tag, MPI_Comm comm){
    CPUIrecv<complexFloatDevice>* out = new CPUIrecv<complexFloatDevice>(buff,n,source,tag,comm);
    return out;
}
#endif


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

#ifndef SWFFT_NOCUDAMPI
template<class T>
void cpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm){
    int world_size; MPI_Comm_size(comm,&world_size);
    int sz = world_size * n * sizeof(T);
    T* d_buff1; gpuMalloc(&d_buff1,sz);
    T* d_buff2; gpuMalloc(&d_buff2,sz);
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
#ifndef SWFFT_NOCUDAMPI

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

void GPUMPI::query(){
    printf("Using GPUMPI\n");
}

void GPUMPI::sendrecv(complexDoubleDevice* send_buff, int sendcount, int dest, int sendtag, complexDoubleDevice* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    base_sendrecv(send_buff,sendcount,dest,sendtag,recv_buff,recvcount,source,recvtag,comm);
}

void GPUMPI::sendrecv(complexFloatDevice* send_buff, int sendcount, int dest, int sendtag, complexFloatDevice* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    base_sendrecv(send_buff,sendcount,dest,sendtag,recv_buff,recvcount,source,recvtag,comm);
}

void GPUMPI::sendrecv(complexDoubleHost* send_buff, int sendcount, int dest, int sendtag, complexDoubleHost* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    size_t send_size = sendcount * sizeof(complexDoubleHost);
    size_t recv_size = recvcount * sizeof(complexDoubleHost);
    complexDoubleDevice* h_buff1; swfftAlloc(&h_buff1,send_size);
    complexDoubleDevice* h_buff2; swfftAlloc(&h_buff2,send_size);
    gpuMemcpy(h_buff1,send_buff,send_size,gpuMemcpyHostToDevice);
    base_sendrecv(h_buff1,sendcount,dest,sendtag,h_buff2,recvcount,source,recvtag,comm);
    gpuMemcpy(recv_buff,h_buff2,recv_size,gpuMemcpyDeviceToHost);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

void GPUMPI::sendrecv(complexFloatHost* send_buff, int sendcount, int dest, int sendtag, complexFloatHost* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm){
    size_t send_size = sendcount * sizeof(complexFloatHost);
    size_t recv_size = recvcount * sizeof(complexFloatHost);
    complexFloatDevice* h_buff1; swfftAlloc(&h_buff1,send_size);
    complexFloatDevice* h_buff2; swfftAlloc(&h_buff2,send_size);
    gpuMemcpy(h_buff1,send_buff,send_size,gpuMemcpyHostToDevice);
    base_sendrecv(h_buff1,sendcount,dest,sendtag,h_buff2,recvcount,source,recvtag,comm);
    gpuMemcpy(recv_buff,h_buff2,recv_size,gpuMemcpyDeviceToHost);
    swfftFree(h_buff1);
    swfftFree(h_buff2);
}

GPUIsend<complexDoubleHost>* GPUMPI::isend(complexDoubleHost* buff, int n, int dest, int tag, MPI_Comm comm){
    GPUIsend<complexDoubleHost>* out = new GPUIsend<complexDoubleHost>(buff,n,dest,tag,comm);
    return out;
}

GPUIsend<complexFloatHost>* GPUMPI::isend(complexFloatHost* buff, int n, int dest, int tag, MPI_Comm comm){
    GPUIsend<complexFloatHost>* out = new GPUIsend<complexFloatHost>(buff,n,dest,tag,comm);
    return out;
}

GPUIrecv<complexDoubleHost>* GPUMPI::irecv(complexDoubleHost* buff, int n, int source, int tag, MPI_Comm comm){
    GPUIrecv<complexDoubleHost>* out = new GPUIrecv<complexDoubleHost>(buff,n,source,tag,comm);
    return out;
}

GPUIrecv<complexFloatHost>* GPUMPI::irecv(complexFloatHost* buff, int n, int source, int tag, MPI_Comm comm){
    GPUIrecv<complexFloatHost>* out = new GPUIrecv<complexFloatHost>(buff,n,source,tag,comm);
    return out;
}

//#ifdef SWFFT_GPU
GPUIsend<complexDoubleDevice>* GPUMPI::isend(complexDoubleDevice* buff, int n, int dest, int tag, MPI_Comm comm){
    //printf("make isend!\n");
    GPUIsend<complexDoubleDevice>* out = new GPUIsend<complexDoubleDevice>(buff,n,dest,tag,comm);
    return out;
}

GPUIsend<complexFloatDevice>* GPUMPI::isend(complexFloatDevice* buff, int n, int dest, int tag, MPI_Comm comm){
    GPUIsend<complexFloatDevice>* out = new GPUIsend<complexFloatDevice>(buff,n,dest,tag,comm);
    return out;
}

GPUIrecv<complexDoubleDevice>* GPUMPI::irecv(complexDoubleDevice* buff, int n, int source, int tag, MPI_Comm comm){
    //printf("make irecv\n");
    GPUIrecv<complexDoubleDevice>* out = new GPUIrecv<complexDoubleDevice>(buff,n,source,tag,comm);//(buff,n,source,tag,comm);
    return out;
}

GPUIrecv<complexFloatDevice>* GPUMPI::irecv(complexFloatDevice* buff, int n, int source, int tag, MPI_Comm comm){
    GPUIrecv<complexFloatDevice>* out = new GPUIrecv<complexFloatDevice>(buff,n,source,tag,comm);
    return out;
}
//#endif

#endif
#endif
}