#include "gpu.hpp"
#include "mpiwrangler.hpp"
#include <stdlib.h>
namespace SWFFT{

    template<class T>
    CPUIsend<T>::CPUIsend(T* in_buff, int n, int dest, int tag, MPI_Comm comm){
        MPI_Isend(in_buff,n * sizeof(T),MPI_BYTE,dest,tag,comm,&req);
    }
    
    template<class T>
    CPUIsend<T>::~CPUIsend(){
        
    }

    template<class T>
    void CPUIsend<T>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    #ifdef SWFFT_GPU
    template<>
    CPUIsend<complexDoubleDevice>::CPUIsend(complexDoubleDevice* in_buff, int n, int dest, int tag, MPI_Comm comm){
        h_in_buff = malloc(sizeof(complexDoubleDevice) * n);
        gpuMemcpy(h_in_buff,in_buff,sizeof(n) * sizeof(complexDoubleDevice),gpuMemcpyDeviceToHost);
        MPI_Isend(h_in_buff,n * sizeof(complexDoubleDevice),MPI_BYTE,dest,tag,comm,&req);
    }

    template<>
    CPUIsend<complexDoubleDevice>::~CPUIsend(){}

    template<>
    void CPUIsend<complexDoubleDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        free(h_in_buff);
    }

    template<>
    CPUIsend<complexFloatDevice>::CPUIsend(complexFloatDevice* in_buff, int n, int dest, int tag, MPI_Comm comm){
        h_in_buff = malloc(sizeof(complexFloatDevice) * n);
        gpuMemcpy(h_in_buff,in_buff,sizeof(n) * sizeof(complexFloatDevice),gpuMemcpyDeviceToHost);
        MPI_Isend(h_in_buff,n * sizeof(complexFloatDevice),MPI_BYTE,dest,tag,comm,&req);
    }

    template<>
    CPUIsend<complexFloatDevice>::~CPUIsend(){}

    template<>
    void CPUIsend<complexFloatDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        free(h_in_buff);
    }
    #endif

    template<class T>
    CPUIrecv<T>::CPUIrecv(T* my_out_buff, int n, int source, int tag, MPI_Comm comm){
        MPI_Irecv(my_out_buff,n * sizeof(T),MPI_BYTE,source,tag,comm,&req);
    }
    
    template<class T>
    CPUIrecv<T>::~CPUIrecv(){
        
    }

    template<class T>
    void CPUIrecv<T>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    #ifdef SWFFT_GPU
    template<>
    CPUIrecv<complexDoubleDevice>::CPUIrecv(complexDoubleDevice* my_out_buff, int n, int source, int tag, MPI_Comm comm){
        sz = n * sizeof(complexDoubleDevice);
        h_out_buff = malloc(sz);
        out_buff = my_out_buff;
        MPI_Irecv(h_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    CPUIrecv<complexDoubleDevice>::~CPUIrecv(){}

    template<>
    void CPUIrecv<complexDoubleDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuMemcpy(out_buff,h_out_buff,sz,gpuMemcpyHostToDevice);
        free(h_out_buff);
    }

    template<>
    CPUIrecv<complexFloatDevice>::CPUIrecv(complexFloatDevice* my_out_buff, int n, int source, int tag, MPI_Comm comm){
        sz = n * sizeof(complexFloatDevice);
        h_out_buff = malloc(sz);
        out_buff = my_out_buff;
        MPI_Irecv(h_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    CPUIrecv<complexFloatDevice>::~CPUIrecv(){}

    template<>
    void CPUIrecv<complexFloatDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuMemcpy(out_buff,h_out_buff,sz,gpuMemcpyHostToDevice);
        free(h_out_buff);
    }
    #endif

    template class CPUIsend<complexDoubleHost>;
    template class CPUIsend<complexFloatHost>;
    template class CPUIrecv<complexDoubleHost>;
    template class CPUIrecv<complexFloatHost>;
    #ifdef SWFFT_GPU
    template class CPUIsend<complexDoubleDevice>;
    template class CPUIsend<complexFloatDevice>;
    template class CPUIrecv<complexDoubleDevice>;
    template class CPUIrecv<complexFloatDevice>;
    #endif
}