#ifdef SWFFT_HQFFT

#include "hqfft.hpp"

namespace SWFFT{
namespace HQFFT{

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::Dfft(Dist<CollectiveComm,MPI_T,REORDER_T>& dist_, bool k_in_blocks_) : dist(dist_), ng{dist.ng[0],dist.ng[1],dist.ng[2]}, nlocal(dist.nlocal), k_in_blocks(k_in_blocks_){

    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::~Dfft(){
        
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    template<class T>
    inline void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::_forward(T* buff1, T* buff2){

        dist.pencils_1(buff1,buff2);

        FFTs.forward(buff1,buff2,ng[2],nlocal/ng[2]);

        dist.pencils_2(buff2,buff1);

        FFTs.forward(buff1,buff2,ng[1],nlocal/ng[1]);

        dist.pencils_3(buff2,buff1);

        FFTs.forward(buff1,buff2,ng[0],nlocal/ng[0]);

        if (k_in_blocks){

            dist.return_pencils(buff1,buff2);

        } else {

            copyBuffers<T> cpy(buff1,buff2,nlocal);
            cpy.wait();

        }

    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    template<class T>
    inline void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::_backward(T* buff1, T* buff2){
        if(k_in_blocks){
            dist.pencils_1(buff1,buff2);

            FFTs.backward(buff1,buff2,ng[2],nlocal/ng[2]);

            dist.pencils_2(buff2,buff1);

            FFTs.backward(buff1,buff2,ng[1],nlocal/ng[1]);

            dist.pencils_3(buff2,buff1);

            FFTs.backward(buff1,buff2,ng[0],nlocal/ng[0]);

            dist.return_pencils(buff1,buff2);
        } else {
            
            FFTs.backward(buff1,buff2,ng[0],nlocal/ng[0]);

            dist.inverse_pencils_3(buff2, buff1);

            FFTs.backward(buff1,buff2,ng[1],nlocal/ng[1]);

            dist.inverse_pencils_2(buff2,buff1);

            FFTs.backward(buff1,buff2,ng[2],nlocal/ng[2]);

            dist.inverse_pencils_1(buff2,buff1);

            //copyBuffers<T> cpy(buff1,buff2,nlocal);
            //cpy.wait();

        }
    }
    
    #ifdef SWFFT_GPU
    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::forward(complexDoubleDevice* buff1, complexDoubleDevice* buff2){
        _forward(buff1,buff2);
        gpuDeviceSynchronize();
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::forward(complexFloatDevice* buff1, complexFloatDevice* buff2){
        _forward(buff1,buff2);
        gpuDeviceSynchronize();
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::backward(complexDoubleDevice* buff1, complexDoubleDevice* buff2){
        _backward(buff1,buff2);
        gpuDeviceSynchronize();
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::backward(complexFloatDevice* buff1, complexFloatDevice* buff2){
        _backward(buff1,buff2);
        gpuDeviceSynchronize();
    }
    #endif

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::forward(complexDoubleHost* buff1, complexDoubleHost* buff2){
        _forward(buff1,buff2);
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::forward(complexFloatHost* buff1, complexFloatHost* buff2){
        _forward(buff1,buff2);
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::backward(complexDoubleHost* buff1, complexDoubleHost* buff2){
        _backward(buff1,buff2);
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    void Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::backward(complexFloatHost* buff1, complexFloatHost* buff2){
        _backward(buff1,buff2);
    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    int Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::buff_sz(){
        return nlocal;
    }
    #ifdef SWFFT_GPU
    #ifdef SWFFT_CUFFT
    template class Dfft<Distribution,GPUReshape,AllToAll,CPUMPI,gpuFFT>;
    template class Dfft<Distribution,GPUReshape,PairSends,CPUMPI,gpuFFT>;
    template class Dfft<Distribution,CPUReshape,AllToAll,CPUMPI,gpuFFT>;
    template class Dfft<Distribution,CPUReshape,PairSends,CPUMPI,gpuFFT>;
    #endif
    #ifdef SWFFT_FFTW
    template class Dfft<Distribution,GPUReshape,AllToAll,CPUMPI,fftw>;
    template class Dfft<Distribution,GPUReshape,PairSends,CPUMPI,fftw>;
    #endif
    #endif

    #ifdef SWFFT_FFTW
    template class Dfft<Distribution,CPUReshape,AllToAll,CPUMPI,fftw>;
    template class Dfft<Distribution,CPUReshape,PairSends,CPUMPI,fftw>;
    #endif

}
}
#endif