#ifndef SWFFT_SEEN
#define SWFFT_SEEN
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "gpu.hpp"
#include "complex-type.h"
#include "timing-stats.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SWFFT_ALLTOALL
#include "alltoall.hpp"
#endif

#ifdef SWFFT_PAIRWISE
#include "pairwise.hpp"
#endif

namespace SWFFT{

    inline int swfft_init_threads(int nthreads = 0){
        int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);

        #ifdef _OPENMP
        int omt = omp_get_max_threads();
        #endif

        #ifdef SWFFT_FFTW
        int out = 1;
        #ifdef _OPENMP
        if (nthreads != 0){
            out = swfft_fftw_init_threads(nthreads);
        } else {
            out = swfft_fftw_init_threads(omt);
        }
        
        #endif
        if (rank == 0){
            printf("swfft::fftw initialized with %d threads\n",out);
        }
        return out;
        #endif

        return 0;
        
    }

    template<template<class,class>class DistBackend, class MPI_T, class FFTBackend>
    class swfft{
        private:
            DistBackend<MPI_T,FFTBackend> backend;
            double last_time;
            int last_was;
        
        public:
            inline swfft(MPI_Comm comm, int ngx, int blockSize = 64, bool ks_as_block = true) : backend(comm,ngx,blockSize,ks_as_block), last_time(0), last_was(-1){

            }

            inline swfft(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize = 64, bool ks_as_block = true) : backend(comm,ngx,ngy,ngz,blockSize,ks_as_block), last_time(0), last_was(-1){

            }

            inline void printLastTime(){
                if (last_was == 0){
                    printTimingStats(backend.comm(),"FORWARD ",last_time);
                } else {
                    printTimingStats(backend.comm(),"BACKWARD",last_time);
                }
                
            }

            inline void query(){
                int world_rank; MPI_Comm_rank(backend.comm(),&world_rank);
                int world_size; MPI_Comm_size(backend.comm(),&world_size);
                if (world_rank == 0){
                    printf("swfft configured:\n   ");
                    backend.query();
                    printf("   n = [%d %d %d]\n",ngx(),ngy(),ngz());
                    printf("   world_size = %d\n",world_size);
                }
            }

            inline bool test_distribution(){
                return backend.test_distribution();
            }

            inline int3 get_ks(int idx){
                return backend.get_ks(idx);
            }

            inline int ngx(){
                return backend.ngx();
            }

            inline int ngy(){
                return backend.ngy();
            }

            inline int ngz(){
                return backend.ngz();
            }

            inline int3 ng(){
                return backend.ng();
            }

            inline int ng(int i){
                return backend.ng(i);
            }

            inline int buff_sz(){
                return backend.buff_sz();
            }

            inline int3 coords(){
                return backend.coords();
            }

            inline int rank(){
                return backend.rank();
            }

            inline MPI_Comm comm(){
                return backend.comm();
            }

            template<class T>
            inline void forward(T* buff1, T* buff2){
                double start = MPI_Wtime();

                backend.forward(buff1,buff2);

                double end = MPI_Wtime();
                last_time = end-start;
                last_was = 0;
            }

            template<class T>
            inline void backward(T* buff1, T* buff2){
                double start = MPI_Wtime();

                backend.backward(buff1,buff2);

                double end = MPI_Wtime();
                last_time = end-start;
                last_was = 1;
            }

            template<class T>
            inline void forward(T* buff1){
                double start = MPI_Wtime();

                backend.forward(buff1);

                double end = MPI_Wtime();
                last_time = end-start;
                last_was = 0;
            }

            template<class T>
            inline void backward(T* buff1){
                double start = MPI_Wtime();

                backend.backward(buff1);

                double end = MPI_Wtime();
                last_time = end-start;
                last_was = 1;
            }

    };

}
#endif