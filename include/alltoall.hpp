#ifndef ALLTOALLSEEN
#define ALLTOALLSEEN

#ifdef ALLTOALL

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "fftinterface.hpp"
#include "backend.hpp"

namespace A2A{

    template<class T>
    class Distribution{
        public:
            int ndims;
            int Ng;
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

            Distribution(MPI_Comm input_comm, int input_Ng, int input_blockSize, int nBatches);

            ~Distribution();

            void memcpy_d2h(T* h, T* d, int batch, gpuStream_t stream);
            void memcpy_h2d(T* d, T* h, int batch, gpuStream_t stream);

            MPI_Comm shuffle_comm_1();
            MPI_Comm shuffle_comm_2();
            MPI_Comm shuffle_comm(int n);

            void shuffle_indices(T* Buff1, T* Buff2, int n);

            void getPencils(T* Buff1, T* Buff2, int dim, int batch);

            void returnPencils(T* Buff1, T* Buff2, int dim, int batch);

            void reorder(T* Buff1, T* Buff2, int n, int direction, int batch, gpuStream_t stream);

            void s_alltoall_forward(T* Buff1, T* Buff2, int n, int pencil_size, MPI_Comm comm_);

            void s_alltoall_backward(T* Buff1, T* Buff2, int n, int pencil_size, MPI_Comm comm_);

            void finalize();
    };

    template<class T, class FFTBackend>
    class Dfft {

        private:
            inline void fft(T* d_data, int direction);
        
        public:
            int Ng;
            int nlocal;
            int world_size;
            int blockSize;
            Distribution<T>& distribution;

            FFTBackend FFTs;

            gpuStream_t fftstream;
            gpuEvent_t* fft_events;
            bool PlansMade;

            T* scratch;
            T* data;

            Dfft(Distribution<T>& dist_);
            ~Dfft();

            void makePlans(T* data_, T* scratch_);
            void makePlans(T* scratch_);
            void makePlans();

            void forward(T* data_);
            void forward();
            void backward(T* data_);
            void backward();

            void finalize();

        
    };

}

template<class T, class FFTBackend>
class AllToAll : public SwfftBackend<T, FFTBackend>{
    public:
        A2A::Distribution<T> dist;
        A2A::Dfft<T,FFTBackend> dfft;

        AllToAll(){};
        AllToAll(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        AllToAll(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        AllToAll(int ngx, int ngy, int ngz, MPI_Comm comm);
        AllToAll(int ng, int blockSize, MPI_Comm comm);
        AllToAll(int ng, MPI_Comm comm);

        ~AllToAll(){};

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

#endif

#endif