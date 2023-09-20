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
        swfft(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true) : backend(comm,ngx,blockSize,ks_as_block){

        }

        swfft(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block = true) : backend(comm,ngx,ngx,ngx,blockSize,ks_as_block){

        }

        bool test_distribution(){
            return backend.test_distribution();
        }

        int ngx(){
            return backend.ngx();
        }

        int ngy(){
            return backend.ngy();
        }

        int ngz(){
            return backend.ngz();
        }

        int3 ng(){
            return backend.ng();
        }

        int ng(int i){
            return backend.ng(i);
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