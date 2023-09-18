#ifndef SWFFTSEEN
#define SWFFTSEEN

#include <type_traits>

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "alltoall.hpp"
#include "pairwise.hpp"
#include "complex-type.h"

#include <iostream>

#include <cassert>

/*class ctCounter {
public:
    ctCounter()
        : value{ 0 }
    {
    }
    int accumulate(int value_) {
        return (value += value_), value;
    }
    int value;

    constexpr ctCounter() : value{ 0 }{}

    constexpr int accumulate(int value_) {
        value += value_;
        return value;
    }

};*/


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

void swfftAlloc(complexDoubleHost** ptr, size_t sz){
    *ptr = (complexDoubleHost*)malloc(sz);
}
void swfftAlloc(complexFloatHost** ptr, size_t sz){
    *ptr = (complexFloatHost*)malloc(sz);
}
void swfftFree(complexDoubleHost* ptr){
    free(ptr);
}
void swfftFree(complexFloatHost* ptr){
    free(ptr);
}

template<class T>
class Caster{
    public:
        T* buff;
        void* original;
        int allocated;
        int sz;
        Caster(complexDoubleDevice* in, int buffsz);
        Caster(complexFloatDevice* in, int buffsz);
        Caster(complexDoubleHost* in, int buffsz);
        Caster(complexFloatHost* in, int buffsz);
        ~Caster();
        T* get(){return buff;}
};

template<>
Caster<complexDoubleDevice>::Caster(complexDoubleDevice* in, int buffsz){
    buff = in;
    allocated = 0;
}

template<>
Caster<complexDoubleDevice>::Caster(complexDoubleHost* in, int buffsz){
    original = in;
    sz = buffsz;
    swfftAlloc(&buff,sizeof(complexDoubleDevice) * buffsz);
    gpuMemcpy(buff,in,sizeof(complexDoubleDevice) * buffsz,gpuMemcpyHostToDevice);
    allocated = 1;
}

template<>
Caster<complexDoubleDevice>::~Caster(){
    if (allocated){
        gpuMemcpy(original,buff,sizeof(complexDoubleDevice) * sz,gpuMemcpyDeviceToHost);
        gpuFree(buff);
    }
}

template<template<class,template<class> class> class Backend, template<class> class FFTBackend, class T>
class swfft{
    private:
        Backend<T,FFTBackend> backend;
        //constexpr ctCounter which;
        int plans_type;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,batches,comm), plans_type(0){};
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,comm), plans_type(0){};
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm) : backend(ngx,ngy,ngz,comm), plans_type(0){};
        swfft(int ng, int blockSize, MPI_Comm comm) : backend(ng,blockSize,comm), plans_type(0){};
        swfft(int ng, MPI_Comm comm) : backend(ng,comm), plans_type(0){};
        
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

        void assignDelta(complexDoubleHost* buff){
            for (int i = 0; i < backend.buffsz(); i++){
                buff[i] = complexDoubleHost(0,0);
            }
            //gpuMemset(buff,0,sizeof(complexDoubleDevice) * backend.buffsz());
            int3 my_coords = coords();
            if ((my_coords.x == 0) && (my_coords.y == 0) && (my_coords.z == 0)){
                buff[0] = complexDoubleHost(1,0);
            }
        }

        void assignDelta(complexFloatHost* buff){
            for (int i = 0; i < backend.buffsz(); i++){
                buff[i] = complexFloatHost(0,0);
            }
            //gpuMemset(buff,0,sizeof(complexDoubleDevice) * backend.buffsz());
            int3 my_coords = coords();
            if ((my_coords.x == 0) && (my_coords.y == 0) && (my_coords.z == 0)){
                buff[0] = complexFloatHost(1,0);
            }
        }
        
        void makePlans(T* buff1, T* buff2){
            //static_assert(which.value == 0, "Plans can only be made once!");
            //which.accumulate(1);
            assert(plans_type == 0);
            plans_type = 1;
            backend.makePlans(buff1,buff2);
        };
        
        void makePlans(T* buff2){
            assert(plans_type == 0);
            plans_type = 2;
            backend.makePlans(buff2);
        };

        void forward(){
            assert(plans_type == 1);
            backend.forward();
        };

        template<class T1>
        void forward(T1* buff1){
            assert(plans_type != 0);
            Caster<T> cast(buff1,buffsz());
            backend.forward(cast.buff);
        };

        void backward(){
            assert(plans_type == 1);
            backend.backward();
        };

        template<class T1>
        void backward(T1* buff1){
            assert(plans_type != 0);
            Caster<T> cast(buff1,buffsz());
            backend.backward(cast.buff);
        };

};

#endif