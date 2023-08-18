#ifndef PAIRWISESEEN
#define PAIRWISESEEN

#ifdef PAIRWISE

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "fftinterface.hpp"
#include "backend.hpp"


template<class T, class FFTBackend>
class Pairwise : public SwfftBackend<T,FFTBackend>{
    public:
        Pairwise();
        Pairwise(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        Pairwise(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        Pairwise(int ngx, int ngy, int ngz, MPI_Comm comm);
        Pairwise(int ng, int blockSize, MPI_Comm comm);
        Pairwise(int ng, MPI_Comm comm);

        ~Pairwise();

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