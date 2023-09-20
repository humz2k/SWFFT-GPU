#ifndef SWFFT_SEEN
#define SWFFT_SEEN
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "gpu.hpp"
#include "complex-type.h"

#ifdef ALLTOALL
#include "alltoall.hpp"
#endif

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
class swfft{
    private:
        DistBackend<MPI_T,FFTBackend> backend;
    
    public:
        swfft(MPI_Comm comm, int ngx, int blockSize) : backend(comm,ngx,blockSize){

        }

        swfft(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize) : backend(comm,ngx,ngx,ngx,blockSize){

        }

        int buff_sz(){
            return backend.buff_sz();
        }

        int3 coords(){
            return backend.coords();
        }

        int rank(){
            return backend.rank();
        }

        MPI_Comm comm(){
            return backend.comm();
        }

        template<class T>
        void forward(T* buff1, T* buff2){
            backend.forward(buff1,buff2);
        }

        template<class T>
        void backward(T* buff1, T* buff2){
            backend.backward(buff1,buff2);
        }

        template<class T>
        void forward(T* buff1){
            backend.forward(buff1);
        }

        template<class T>
        void backward(T* buff1){
            backend.backward(buff1);
        }

};
#endif