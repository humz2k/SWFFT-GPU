#include "gpu.hpp"
#include "mpiwrangler.hpp"
#include <stdlib.h>
namespace SWFFT{

    template<class T>
    CPUIsend<T>::CPUIsend() : initialized(false){
        
    }

    template<class T>
    CPUIsend<T>::CPUIsend(T* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        //printf("WTF\n");
    }
    
    template<class T>
    CPUIsend<T>::~CPUIsend(){
        
    }

    template<class T>
    void CPUIsend<T>::execute(){
        //printf("WTF1\n");
        MPI_Isend(in_buff,n * sizeof(T),MPI_BYTE,dest,tag,comm,&req);
    }

    template<class T>
    void CPUIsend<T>::wait(){
        //printf("WTF2\n");
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    #ifdef SWFFT_GPU
    template<>
    CPUIsend<complexDoubleDevice>::CPUIsend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        size_t sz = sizeof(complexDoubleDevice) * n;        
        h_in_buff = malloc(sz);
        gpuEventCreate(&event);
        gpuMemcpyAsync(h_in_buff,in_buff,sz,gpuMemcpyDeviceToHost);
        gpuEventRecord(event);
    }

    template<>
    CPUIsend<complexDoubleDevice>::CPUIsend() : initialized(false){
        
    }

    template<>
    CPUIsend<complexFloatDevice>::CPUIsend() : initialized(false){
        
    }

    template<>
    CPUIsend<complexDoubleDevice>::~CPUIsend(){
        
    }

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
        size_t sz = sizeof(complexFloatDevice) * n;
        h_in_buff = malloc(sz);
        gpuEventCreate(&event);
        gpuMemcpyAsync(h_in_buff,in_buff,n * sizeof(complexFloatDevice),gpuMemcpyDeviceToHost);
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
        //printf("WTF3\n");
    }

    template<class T>
    CPUIrecv<T>::CPUIrecv() : initialized(false){
        
    }
    
    template<class T>
    CPUIrecv<T>::~CPUIrecv(){
        
    }

    template<class T>
    void CPUIrecv<T>::execute(){
        //printf("WTF4\n");
        MPI_Irecv(out_buff,n * sizeof(T),MPI_BYTE,source,tag,comm,&req);
    }

    template<class T>
    void CPUIrecv<T>::wait(){
        //printf("WTF5\n");
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
    CPUIrecv<complexDoubleDevice>::~CPUIrecv(){
        //printf("Irecv::delete!\n");
    }

    template<>
    void CPUIrecv<complexDoubleDevice>::execute(){
        //printf("Irecv::execute!\n");
        MPI_Irecv(h_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    void CPUIrecv<complexDoubleDevice>::wait(){
        //printf("Irecv::wait!\n");
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuEventCreate(&event);
        gpuMemcpyAsync(out_buff,h_out_buff,sz,gpuMemcpyHostToDevice);
        gpuEventRecord(event);
        //free(h_out_buff);
    }

    template<>
    void CPUIrecv<complexDoubleDevice>::finalize(){
        //printf("Irecv::finalize!\n");
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

#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI
namespace SWFFT{

    template<class T>
    GPUIsend<T>::GPUIsend() : initialized(false){
        
    }

    template<class T>
    GPUIsend<T>::GPUIsend(T* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        //printf("WTF\n");
    }
    
    template<class T>
    GPUIsend<T>::~GPUIsend(){
        
    }

    template<class T>
    void GPUIsend<T>::execute(){
        //printf("WTF1\n");
        MPI_Isend(in_buff,n * sizeof(T),MPI_BYTE,dest,tag,comm,&req);
    }

    template<class T>
    void GPUIsend<T>::wait(){
        //printf("WTF2\n");
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    //#ifdef SWFFT_GPU
    template<>
    GPUIsend<complexDoubleHost>::GPUIsend(complexDoubleHost* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        size_t sz = sizeof(complexDoubleDevice) * n;        
        gpuMalloc(&d_in_buff,sz);
        gpuEventCreate(&event);
        gpuMemcpyAsync(d_in_buff,in_buff,sz,gpuMemcpyHostToDevice);
        gpuEventRecord(event);
    }

    template<>
    GPUIsend<complexDoubleHost>::GPUIsend() : initialized(false){
        
    }

    template<>
    GPUIsend<complexFloatHost>::GPUIsend() : initialized(false){
        
    }

    template<>
    GPUIsend<complexDoubleHost>::~GPUIsend(){
        
    }

    template<>
    void GPUIsend<complexDoubleHost>::execute(){
        
        gpuEventSynchronize(event);
        MPI_Isend(d_in_buff,n * sizeof(complexDoubleHost),MPI_BYTE,dest,tag,comm,&req);
        gpuEventDestroy(event);
    }

    template<>
    void GPUIsend<complexDoubleHost>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuFree(d_in_buff);
    }

    template<>
    GPUIsend<complexFloatHost>::GPUIsend(complexFloatHost* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_), comm(comm_){
        size_t sz = sizeof(complexFloatHost) * n;
        gpuMalloc(&d_in_buff,sz);
        gpuEventCreate(&event);
        gpuMemcpyAsync(d_in_buff,in_buff,n * sizeof(complexFloatDevice),gpuMemcpyHostToDevice);
        gpuEventRecord(event);
    }

    template<>
    GPUIsend<complexFloatHost>::~GPUIsend(){}

    template<>
    void GPUIsend<complexFloatHost>::execute(){
        gpuEventSynchronize(event);
        MPI_Isend(d_in_buff,n * sizeof(complexFloatHost),MPI_BYTE,dest,tag,comm,&req);
        gpuEventDestroy(event);
    }

    template<>
    void GPUIsend<complexFloatHost>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuFree(d_in_buff);
    }
    //#endif

    template<class T>
    GPUIrecv<T>::GPUIrecv(T* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_) : initialized(true), out_buff(out_buff_), n(n_), source(source_), tag(tag_), comm(comm_){
        //printf("WTF3\n");
    }

    template<class T>
    GPUIrecv<T>::GPUIrecv() : initialized(false){
        
    }
    
    template<class T>
    GPUIrecv<T>::~GPUIrecv(){
        
    }

    template<class T>
    void GPUIrecv<T>::execute(){
        //printf("WTF4\n");
        MPI_Irecv(out_buff,n * sizeof(T),MPI_BYTE,source,tag,comm,&req);
    }

    template<class T>
    void GPUIrecv<T>::wait(){
        //printf("WTF5\n");
        MPI_Wait(&req,MPI_STATUS_IGNORE);
    }

    template<class T>
    void GPUIrecv<T>::finalize(){
        
    }

    //#ifdef SWFFT_GPU
    template<>
    GPUIrecv<complexDoubleHost>::GPUIrecv(complexDoubleHost* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_) : initialized(true), out_buff(out_buff_), n(n_), source(source_), tag(tag_), comm(comm_){
        sz = n * sizeof(complexDoubleHost);
        gpuMalloc(&d_out_buff,sz);
    }

    template<>
    GPUIrecv<complexDoubleHost>::GPUIrecv() : initialized(false){
        
    }

    template<>
    GPUIrecv<complexFloatHost>::GPUIrecv() : initialized(false){
        
    }

    template<>
    GPUIrecv<complexDoubleHost>::~GPUIrecv(){
        //printf("Irecv::delete!\n");
    }

    template<>
    void GPUIrecv<complexDoubleHost>::execute(){
        //printf("Irecv::execute!\n");
        MPI_Irecv(d_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    void GPUIrecv<complexDoubleHost>::wait(){
        //printf("Irecv::wait!\n");
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuEventCreate(&event);
        gpuMemcpyAsync(out_buff,d_out_buff,sz,gpuMemcpyDeviceToHost);
        gpuEventRecord(event);
        //free(h_out_buff);
    }

    template<>
    void GPUIrecv<complexDoubleHost>::finalize(){
        //printf("Irecv::finalize!\n");
        gpuEventSynchronize(event);
        gpuFree(d_out_buff);
        gpuEventDestroy(event);
    }

    template<>
    GPUIrecv<complexFloatHost>::GPUIrecv(complexFloatHost* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_) : initialized(true), out_buff(out_buff_), n(n_), source(source_), tag(tag_), comm(comm_){
        sz = n * sizeof(complexFloatHost);
        gpuMalloc(&d_out_buff,sz);   
    }

    template<>
    GPUIrecv<complexFloatHost>::~GPUIrecv(){}

    template<>
    void GPUIrecv<complexFloatHost>::execute(){
        MPI_Irecv(d_out_buff,sz,MPI_BYTE,source,tag,comm,&req);
    }

    template<>
    void GPUIrecv<complexFloatHost>::wait(){
        MPI_Wait(&req,MPI_STATUS_IGNORE);
        gpuEventCreate(&event);
        gpuMemcpyAsync(out_buff,d_out_buff,sz,gpuMemcpyDeviceToHost);
        gpuEventRecord(event);
        //free(h_out_buff);
    }

    template<>
    void GPUIrecv<complexFloatHost>::finalize(){
        gpuEventSynchronize(event);
        gpuFree(d_out_buff);
        gpuEventDestroy(event);
    }
    //#endif

    template class GPUIsend<complexDoubleHost>;
    template class GPUIsend<complexFloatHost>;
    template class GPUIrecv<complexDoubleHost>;
    template class GPUIrecv<complexFloatHost>;
    //#ifdef SWFFT_GPU
    template class GPUIsend<complexDoubleDevice>;
    template class GPUIsend<complexFloatDevice>;
    template class GPUIrecv<complexDoubleDevice>;
    template class GPUIrecv<complexFloatDevice>;
    //#endif
}
#endif
#endif