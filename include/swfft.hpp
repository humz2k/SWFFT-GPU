#ifndef SWFFTSEEN
#define SWFFTSEEN

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "alltoall.hpp"
#include "pairwise.hpp"
#include "complex-type.h"

#ifdef GPU
void swfftAlloc(complexDoubleDevice** ptr, size_t sz){
    gpuMalloc(ptr,sz);
}
void swfftAlloc(complexFloatDevice** ptr, size_t sz){
    gpuMalloc(ptr,sz);
}
void swfftFree(complexDoubleDevice* ptr){
    gpuFree(ptr);
}
void swfftFree(complexFloatDevice* ptr){
    gpuFree(ptr);
}
#endif

template<template<class,template<class> class> class Backend, template<class> class FFTBackend, class T>
class swfft{
    private:
        Backend<T,FFTBackend> backend;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,batches,comm){};
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,comm){};
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm) : backend(ngx,ngy,ngz,comm){};
        swfft(int ng, int blockSize, MPI_Comm comm) : backend(ng,blockSize,comm){};
        swfft(int ng, MPI_Comm comm) : backend(ng,comm){};
        
        ~swfft(){};

        int buffsz(){
            return backend.buffsz();
        }

        int3 coords(){
            return backend.coords();
        }

        int rank(){
            return backend.rank();
        }

        MPI_Comm world_comm(){
            return backend.world_comm();
        }

        void assignDelta(complexDoubleDevice* buff){
            gpuMemset(buff,0,sizeof(complexDoubleDevice) * backend.buffsz());
            int3 my_coords = coords();
            if ((my_coords.x == 0) && (my_coords.y == 0) && (my_coords.z == 0)){
                complexDoubleDevice center;
                center.x = 1;
                center.y = 0;
                gpuMemcpy(buff,&center,sizeof(complexDoubleDevice),gpuMemcpyHostToDevice);
            }
        }

        void assignDelta(complexFloatDevice* buff){
            gpuMemset(buff,0,sizeof(complexFloatDevice) * backend.buffsz());
            int3 my_coords = coords();
            if ((my_coords.x == 0) && (my_coords.y == 0) && (my_coords.z == 0)){
                complexFloatDevice center;
                center.x = 1;
                center.y = 0;
                gpuMemcpy(buff,&center,sizeof(complexFloatDevice),gpuMemcpyHostToDevice);
            }
        }

        void makePlans(T* buff1, T* buff2){
            backend.makePlans(buff1,buff2);
        };
        void makePlans(T* buff2){
            backend.makePlans(buff2);
        };

        void forward(){
            backend.forward();
        };
        void forward(T* buff1){
            backend.forward(buff1);
        };
        void backward(){
            backend.backward();
        };
        void backward(T* buff1){
            backend.backward(buff1);
        };

};

#endif