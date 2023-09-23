#ifdef SWFFT_HQFFT
#ifndef SWFFT_HQFFT_SEEN
#define SWFFT_HQFFT_SEEN

#include <mpi.h>
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "hqfft_reorder.hpp"

namespace SWFFT{

namespace HQFFT{

    template<class MPI_T>
    class CollectiveCommunicator{
        public:
            MPI_T mpi;

            CollectiveCommunicator();
            ~CollectiveCommunicator();

            //template<class T>
            //void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            void query();

    };
    
    template<class MPI_T>
    class AllToAll : public CollectiveCommunicator<MPI_T>{
        private:
            template<class T>
            inline void _alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        public:

            #ifdef SWFFT_GPU
            void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm);
            #endif

            void alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm);

            void query();
    };

    template<class MPI_T>
    class PairSends : public CollectiveCommunicator<MPI_T>{
        private:
            template<class T>
            inline void _alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        public:
            #ifdef SWFFT_GPU
            void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm);
            #endif

            void alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm);

            void query();
    };

    template<template<class> class Communicator, class MPI_T, class REORDER_T>
    class Distribution{
        private:
            template<class T>
            void _pencils_1(T* buff1, T* buff2);

            template<class T>
            void _pencils_2(T* buff1, T* buff2);

            template<class T>
            void _pencils_3(T* buff1, T* buff2);

            template<class T>
            void _return_pencils(T* buff1, T* buff2);

        public:
            int ng[3];
            int nlocal;
            int world_size;
            int world_rank;
            int local_grid_size[3];
            int dims[3];
            int coords[3];
            int local_coords_start[3];
            MPI_Comm world_comm;
            MPI_Comm distcomms[4];

            Communicator<MPI_T> CollectiveComm;
            REORDER_T reorder;

            int blockSize;

            Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
            ~Distribution();

            void pencils_1(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void pencils_1(complexFloatHost* buff1, complexFloatHost* buff2);
            void pencils_2(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void pencils_2(complexFloatHost* buff1, complexFloatHost* buff2);
            void pencils_3(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void pencils_3(complexFloatHost* buff1, complexFloatHost* buff2);
            void return_pencils(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void return_pencils(complexFloatHost* buff1, complexFloatHost* buff2);

            #ifdef SWFFT_GPU
            void pencils_1(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void pencils_1(complexFloatDevice* buff1, complexFloatDevice* buff2);
            void pencils_2(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void pencils_2(complexFloatDevice* buff1, complexFloatDevice* buff2);
            void pencils_3(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void pencils_3(complexFloatDevice* buff1, complexFloatDevice* buff2);
            void return_pencils(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void return_pencils(complexFloatDevice* buff1, complexFloatDevice* buff2);
            #endif

            template<class T>
            void reshape_1(T* buff1, T* buff2);

            template<class T>
            void unreshape_1(T* buff1, T* buff2);

            template<class T>
            void reshape_2(T* buff1, T* buff2);

            template<class T>
            void unreshape_2(T* buff1, T* buff2);

            template<class T>
            void reshape_3(T* buff1, T* buff2);

            template<class T>
            void unreshape_3(T* buff1, T* buff2);

            template<class T>
            void reshape_final(T* buff1, T* buff2, int ny, int nz);

            //template<class T>
            //void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            int buff_sz();
    };

    template<template<template<class> class,class> class Dist, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    class Dfft{
        public:
            Dist<CollectiveComm,MPI_T>& dist;
            FFTBackend FFTs;
            int ng;
            int nlocal;
            
            #ifdef SWFFT_GPU
            void forward(complexDoubleDevice* data, complexDoubleDevice* scratch);
            void forward(complexFloatDevice* data, complexFloatDevice* scratch);
            void backward(complexDoubleDevice* data, complexDoubleDevice* scratch);
            void backward(complexFloatDevice* data, complexFloatDevice* scratch);
            #endif

            void forward(complexDoubleHost* data, complexDoubleHost* scratch);
            void forward(complexFloatHost* data, complexFloatHost* scratch);
            void backward(complexDoubleHost* data, complexDoubleHost* scratch);
            void backward(complexFloatHost* data, complexFloatHost* scratch);
    };

}

}

#endif
#endif