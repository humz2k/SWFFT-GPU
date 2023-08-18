
#ifndef BACKENDSEEN
#define BACKENDSEEN

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

template<class T, class FFTBackend>
class SwfftBackend{
    public:
        SwfftBackend();
        SwfftBackend(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        SwfftBackend(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        SwfftBackend(int ngx, int ngy, int ngz, MPI_Comm comm);
        SwfftBackend(int ng, int blockSize, MPI_Comm comm);
        SwfftBackend(int ng, MPI_Comm comm);

        ~SwfftBackend();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

#endif