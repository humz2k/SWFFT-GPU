#include <mpi.h>
#include "swfft.hpp"

void test(){

    swfft<AllToAll,GPUFFT,complexDouble>(8,MPI_COMM_WORLD);

}

int main(){

    MPI_Init(NULL,NULL);

    test();

    MPI_Finalize();

}