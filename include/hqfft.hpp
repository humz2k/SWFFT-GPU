#ifdef SWFFT_HQFFT
#ifndef SWFFT_HQFFT_SEEN
#define SWFFT_HQFFT_SEEN

#include <mpi.h>
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"

namespace SWFFT{

namespace HQFFT{

    template<class MPI_T>
    class CollectiveCommunicator{
        public:
            MPI_T mpi;

            CollectiveCommunicator();
            ~CollectiveCommunicator();

            template<class T>
            void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            void query();

    };
    
    template<class MPI_T>
    class AllToAll : public CollectiveCommunicator<MPI_T>{
        public:

            template<class T>
            void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            void query();
    };

    template<class MPI_T>
    class PairSends : public CollectiveCommunicator<MPI_T>{
        public:
            template<class T>
            void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            void query();
    };

    template<template<class> class Communicator, class MPI_T>
    class Distribution{
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

            Communicator<MPI> CollectiveComm;

            int blockSize;

            Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
            ~Distribution();

            template<class T>
            void pencils_1(T* buff1, T* buff2);

            template<class T>
            void pencils_2(T* buff1, T* buff2);

            template<class T>
            void pencils_3(T* buff1, T* buff2);

            template<class T>
            void return_pencils(T* buff1, T* buff2);

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

            template<class T>
            void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            int buff_sz();
    };

    template<template<class,class> class Dist, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
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