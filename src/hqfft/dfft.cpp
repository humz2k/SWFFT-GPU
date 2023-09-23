#ifdef SWFFT_HQFFT

#include "hqfft.hpp"

namespace SWFFT{
namespace HQFFT{

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::Dfft(Dist<CollectiveComm,MPI_T,REORDER_T>& dist_) : dist(dist_), ng{dist.ng[0],dist.ng[1],dist.ng[2]}, nlocal(dist.nlocal){

    }

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    Dfft<Dist,REORDER_T,CollectiveComm,MPI_T,FFTBackend>::~Dfft(){
        
    }

}
}
#endif