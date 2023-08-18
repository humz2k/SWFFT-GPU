#include <mpi.h>
#include "swfft.hpp"
#include <string.h>
#include <iostream>
#include "check_kspace.hpp"

template<template<class,template<class> class> class Backend, template<class> class FFTBackend, class T>
void test(){

    swfft<Backend,FFTBackend,T> fft(8,8,8,MPI_COMM_WORLD);

    complexDoubleDevice* buff1; swfftAlloc(&buff1,sizeof(T) * fft.buffsz());
    complexDoubleDevice* buff2; swfftAlloc(&buff2,sizeof(T) * fft.buffsz());

    fft.makePlans(buff2);

    //int3 coords = fft.coords();
    //printf("rank %d coords: %d %d %d\n",fft.rank(),coords.x,coords.y,coords.z);
    
    fft.assignDelta(buff1);

    //printf("%g %g\n",buff1[0].real(),buff1[0].imag());

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