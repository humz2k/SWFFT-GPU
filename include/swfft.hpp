#ifndef SWFFT_SEEN
#define SWFFT_SEEN
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "gpu.hpp"
#include "complex-type.h"
#include "timing-stats.h"

#ifdef ALLTOALL
#include "alltoall.hpp"
#endif

#ifdef PAIRWISE
#include "pairwise.hpp"
#endif

template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
class swfft{
    private:
        DistBackend<MPI_T,FFTBackend> backend;
        double last_time;
        int last_was;
    
    public:
        swfft(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true) : backend(comm,ngx,blockSize,ks_as_block), last_time(0), last_was(-1){

        }

        swfft(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block = true) : backend(comm,ngx,ngx,ngx,blockSize,ks_as_block), last_time(0), last_was(-1){

        }

        void printLastTime(){
            if (last_was == 0){
                printTimingStats(backend.comm(),"FORWARD ",last_time);
            } else {
                printTimingStats(backend.comm(),"BACKWARD",last_time);
            }
            
        }

        bool test_distribution(){
            return backend.test_distribution();
        }

        inline int3 get_ks(int idx){
            return backend.get_ks(idx);
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
            double start = MPI_Wtime();

            backend.forward(buff1,buff2);

            double end = MPI_Wtime();
            last_time = end-start;
            last_was = 0;
        }

        template<class T>
        void backward(T* buff1, T* buff2){
            double start = MPI_Wtime();

            backend.backward(buff1,buff2);

            double end = MPI_Wtime();
            last_time = end-start;
            last_was = 1;
        }

        template<class T>
        void forward(T* buff1){
            double start = MPI_Wtime();

            backend.forward(buff1);

            double end = MPI_Wtime();
            last_time = end-start;
            last_was = 0;
        }

        template<class T>
        void backward(T* buff1){
            double start = MPI_Wtime();

            backend.backward(buff1);

            double end = MPI_Wtime();
            last_time = end-start;
            last_was = 1;
        }

};
#endif