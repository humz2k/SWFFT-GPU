#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

template<class Backend>
class swfft{
    private:
        Backend backend;
        int ngx;
        int ngy;
        int ngz;
        int blockSize;
        MPI_Comm comm;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm);
        swfft(int ng, MPI_Comm comm);
        
        ~swfft();

};