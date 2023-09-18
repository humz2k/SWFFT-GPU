#ifndef ALLTOALLSEEN
#define ALLTOALLSEEN

#ifdef ALLTOALL

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "fftinterface.hpp"
#include "complex-type.h"

namespace A2A{

    template<class T>
    class Distribution{
        public:
            int ndims;
            int Ng;
            int ng[3];
            int nlocal;
            int world_size;
            int world_rank;
            int local_grid_size[3];
            int dims[3];
            int coords[3];
            int local_coordinates_start[3];
            MPI_Datatype TYPE_COMPLEX;
            MPI_Comm comm;
            MPI_Comm fftcomms[3];

            int blockSize;
            int numBlocks;
            int batches;

            #ifdef nocudampi
            T* h_scratch1;
            T* h_scratch2;
            #endif

            gpuStream_t diststream;

            Distribution(MPI_Comm input_comm, int ngx, int input_blockSize, int nBatches);
            Distribution(MPI_Comm input_comm, int ngx, int ngy, int ngz, int input_blockSize, int nBatches);

            ~Distribution();

            void init();

            void memcpy_d2h(T* h, T* d, int batch, gpuStream_t stream);
            void memcpy_h2d(T* d, T* h, int batch, gpuStream_t stream);

            MPI_Comm shuffle_comm_1();
            MPI_Comm shuffle_comm_2();
            MPI_Comm shuffle_comm(int n);

            void shuffle_indices(T* Buff1, T* Buff2, int n);

            void getPencils(T* Buff1, T* Buff2, int dim, int batch);

            void returnPencils(T* Buff1, T* Buff2, int dim, int batch);

            void reorder(T* Buff1, T* Buff2, int n, int direction, int batch, gpuStream_t stream);

            void finalize();
    };

    template<class T, template<class> class FFTBackend>
    class Dfft {

        private:
            inline void fft(T* d_data, fftdirection direction);
        
        public:
            int Ng;
            int ng[3];
            int nlocal;
            int world_size;
            int blockSize;
            Distribution<T>& distribution;

            FFTBackend<T> FFTs;

            gpuStream_t fftstream;
            gpuEvent_t* fft_events;
            bool PlansMade;

            T* scratch;
            T* data;

            Dfft(Distribution<T>& dist_);
            ~Dfft();

            void makePlans(T* data_, T* scratch_);
            void makePlans(T* scratch_);
            //void makePlans();

            void forward(T* data_);
            void forward();
            void backward(T* data_);
            void backward();

            void finalize();

        
    };

}

template<class T, template<class> class FFTBackend>
class AllToAll{
    public:
        A2A::Distribution<T> dist;
        A2A::Dfft<T,FFTBackend> dfft;

        AllToAll(){

        };
        AllToAll(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm) : dist(comm,ngx,ngy,ngz,blockSize,batches), dfft(dist){

        };
        AllToAll(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm) : dist(comm,ngx,ngy,ngz,blockSize,1), dfft(dist){

        };
        AllToAll(int ngx, int ngy, int ngz, MPI_Comm comm) : dist(comm,ngx,ngy,ngz,64,1), dfft(dist){

        };
        AllToAll(int ng, int blockSize, MPI_Comm comm) : dist(comm,ng,blockSize,1), dfft(dist){

        };
        AllToAll(int ng, MPI_Comm comm) : dist(comm,ng,64,1), dfft(dist){

        };

        ~AllToAll(){};

        int buffsz(){
            return dist.nlocal;
        }

        int3 coords(){
            return make_int3(dist.coords[0],dist.coords[1],dist.coords[2]);
        }

        int rank(){
            return dist.world_rank;
        }

        MPI_Comm world_comm(){
            return dist.comm;
        }

        void makePlans(T* buff1, T* buff2){
            dfft.makePlans(buff1,buff2);
        }

        void makePlans(T* buff2){
            dfft.makePlans(buff2);
        }

        void forward(){
            dfft.forward();
        }
        void forward(T* buff1){
            dfft.forward(buff1);
        }
        void backward(){
            dfft.backward();
        }
        void backward(T* buff1){
            dfft.backward(buff1);
        }
};

#endif

#endif