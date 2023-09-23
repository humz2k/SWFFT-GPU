#include "gpu.hpp"
#include "mpiwrangler.hpp"
#include <stdlib.h>
namespace SWFFT{

    template<class T>
    CPUIsend<T>::CPUIsend() : initialized(false){
        
    }

    template<class T>
    CPUIsend<T>::CPUIsend(T* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        
    }
    
    template<class T>
    CPUIsend<T>::~CPUIsend(){
        
    }

    template<class T>
    void CPUIsend<T>::execute(){
        MPI_Isend(in_buff,n * sizeof(T),MPI_BYTE,dest,tag,comm,&req);
    }

    template<class T>
    void CPUIsend<T>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    #ifdef SWFFT_GPU
    template<>
    CPUIsend<complexDoubleDevice>::CPUIsend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        h_in_buff = malloc(sizeof(complexDoubleDevice) * n);
        gpuEventCreate(&event);
        gpuMemcpyAsync(h_in_buff,in_buff,sizeof(n) * sizeof(complexDoubleDevice),gpuMemcpyDeviceToHost);
        gpuEventRecord(event);
    }

    template<>
    CPUIsend<complexDoubleDevice>::CPUIsend() : initialized(false){
        
    }

    template<>
    CPUIsend<complexFloatDevice>::CPUIsend() : initialized(false){
        
    }

    template<>
    CPUIsend<complexDoubleDevice>::~CPUIsend(){}

    template<>
    void CPUIsend<complexDoubleDevice>::execute(){
        gpuEventSynchronize(event);
        MPI_Isend(h_in_buff,n * sizeof(complexDoubleDevice),MPI_BYTE,dest,tag,comm,&req);
        gpuEventDestroy(event);
    }

    template<>
    void CPUIsend<complexDoubleDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        free(h_in_buff);
    }

    template<>
    CPUIsend<complexFloatDevice>::CPUIsend(complexFloatDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        h_in_buff = malloc(sizeof(complexFloatDevice) * n);
        gpuEventCreate(&event);
        gpuMemcpyAsync(h_in_buff,in_buff,sizeof(n) * sizeof(complexFloatDevice),gpuMemcpyDeviceToHost);
        gpuEventRecord(event);
    }

    template<>
    CPUIsend<complexFloatDevice>::~CPUIsend(){}

    template<>
    void CPUIsend<complexFloatDevice>::execute(){
        gpuEventSynchronize(event);
        MPI_Isend(h_in_buff,n * sizeof(complexFloatDevice),MPI_BYTE,dest,tag,comm,&req);
        gpuEventDestroy(event);
    }

    template<>
    void CPUIsend<complexFloatDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        free(h_in_buff);
    }
    #endif

    template<class T>
    CPUIrecv<T>::CPUIrecv(T* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_) : initialized(true), out_buff(out_buff_), n(n_), source(source_), tag(tag_), comm(comm_){
        
    }

    template<class T>
    CPUIrecv<T>::CPUIrecv() : initialized(false){
        
    }
    
    template<class T>
    CPUIrecv<T>::~CPUIrecv(){
        
    }

    template<class T>
    void CPUIrecv<T>::execute(){
        MPI_Irecv(out_buff,n * sizeof(T),MPI_BYTE,source,tag,comm,&req);
    }

    template<class T>
    void CPUIrecv<T>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    template<class T>
    void CPUIrecv<T>::finalize(){
        
    }

    #ifdef SWFFT_GPU
    template<>
    CPUIrecv<complexDoubleDevice>::CPUIrecv(complexDoubleDevice* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_) : initialized(true), out_buff(out_buff_), n(n_), source(source_), tag(tag_), comm(comm_){
        sz = n * sizeof(complexDoubleDevice);
        h_out_buff = malloc(sz);
    }

    template<>
    CPUIrecv<complexDoubleDevice>::CPUIrecv() : initialized(false){
        
    }

    template<>
    CPUIrecv<complexFloatDevice>::CPUIrecv() : initialized(false){
        
    }

    template<>
    CPUIrecv<complexDoubleDevice>::~CPUIrecv(){}

    template<>
    void CPUIrecv<complexDoubleDevice>::execute(){
        MPI_Irecv(h_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    void CPUIrecv<complexDoubleDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuEventCreate(&event);
        gpuMemcpyAsync(out_buff,h_out_buff,sz,gpuMemcpyHostToDevice);
        gpuEventRecord(event);
        //free(h_out_buff);
    }

    template<>
    void CPUIrecv<complexDoubleDevice>::finalize(){
        gpuEventSynchronize(event);
        free(h_out_buff);
        gpuEventDestroy(event);
    }

    template<>
    CPUIrecv<complexFloatDevice>::CPUIrecv(complexFloatDevice* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_) : initialized(true), out_buff(out_buff_), n(n_), source(source_), tag(tag_), comm(comm_){
        sz = n * sizeof(complexFloatDevice);
        h_out_buff = malloc(sz);   
    }

    template<>
    CPUIrecv<complexFloatDevice>::~CPUIrecv(){}

    template<>
    void CPUIrecv<complexFloatDevice>::execute(){
        MPI_Irecv(h_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    void CPUIrecv<complexFloatDevice>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuEventCreate(&event);
        gpuMemcpyAsync(out_buff,h_out_buff,sz,gpuMemcpyHostToDevice);
        gpuEventRecord(event);
        //free(h_out_buff);
    }

    template<>
    void CPUIrecv<complexFloatDevice>::finalize(){
        gpuEventSynchronize(event);
        free(h_out_buff);
        gpuEventDestroy(event);
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