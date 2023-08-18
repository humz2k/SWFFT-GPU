#include <mpi.h>
#include "swfft.hpp"
#include <string.h>
#include <iostream>
#include "check_kspace.hpp"

template<template<class,template<class> class> class Backend, template<class> class FFTBackend, class T>
void test(){

    swfft<Backend,FFTBackend,T> fft(8,MPI_COMM_WORLD);

    T* buff1; swfftAlloc(&buff1,sizeof(T) * fft.buffsz());
    T* buff2; swfftAlloc(&buff2,sizeof(T) * fft.buffsz());

    fft.makePlans(buff1,buff2);

    //int3 coords = fft.coords();
    //printf("rank %d coords: %d %d %d\n",fft.rank(),coords.x,coords.y,coords.z);
    
    fft.assignDelta(buff1);

    fft.forward(buff1);

    check_kspace(fft,buff1);

    swfftFree(buff1);
    swfftFree(buff2);

}

int main(){

    MPI_Init(NULL,NULL);

    #if defined(ALLTOALL) && defined(GPU)
    test<AllToAll,GPUFFT,complexDoubleDevice>();
    //test<AllToAll,GPUFFT,complexFloatDevice>();
    #endif

    MPI_Finalize();

}