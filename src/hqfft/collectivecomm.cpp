#ifdef SWFFT_HQFFT
#include "hqfft.hpp"

namespace SWFFT{
namespace HQFFT{


template<class MPI_T>
CollectiveCommunicator<MPI_T>::CollectiveCommunicator(){

}

template<class MPI_T>
CollectiveCommunicator<MPI_T>::~CollectiveCommunicator(){
    
}

template<class MPI_T>
template<class T>
void AllToAll<MPI_T>::alltoall(T* src, T* dest, int n_recv, MPI_Comm comm){
    
    this->mpi.alltoall(src,dest,n_recv,comm);

}

template<class MPI_T>
void AllToAll<MPI_T>::query(){
    
    printf("CollectiveCommunicator=AllToAll\n");

}

template<class MPI_T>
template<class T>
void PairSends<MPI_T>::alltoall(T* src, T* dest, int n_recv, MPI_Comm comm){
    
    //this->mpi.alltoall(src,dest,n_recv,comm);

}



}
}

#endif