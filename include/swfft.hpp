#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "backend.hpp"
#include "alltoall.hpp"
#include "pairwise.hpp"

template<template<class,class> class Backend, class T, class FFTBackend>
class swfft{
    private:
        Backend<T,FFTBackend> backend;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm);
        swfft(int ng, int blockSize, MPI_Comm comm);
        swfft(int ng, MPI_Comm comm);
        
        ~swfft();

        void makePlans(T* buff1, T* buff2){
            backend.makePlans(buff1,buff2);
        };
        void makePlans(T* buff2){
            backend.makePlans(buff2);
        };
        void makePlans(){
            backend.makePlans();
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