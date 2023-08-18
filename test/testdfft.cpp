#include <mpi.h>
#include "swfft.hpp"

template<template<class,template<class> class> class Backend, template<class> class FFTBackend, class T>
void test(){

    swfft<Backend,FFTBackend,T>(8,MPI_COMM_WORLD);

}

int main(){

    MPI_Init(NULL,NULL);

    #if defined(ALLTOALL) && defined(GPU)
    test<AllToAll,GPUFFT,complexDouble>();
    #endif

    MPI_Finalize();

}