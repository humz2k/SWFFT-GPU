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
inline void AllToAll<MPI_T>::_alltoall(T* src, T* dest, int n_recv, MPI_Comm comm){
    
    this->mpi.alltoall(src,dest,n_recv,comm);

}

template<class MPI_T>
void AllToAll<MPI_T>::query(){
    
    printf("CollectiveCommunicator=AllToAll\n");

}

template<class MPI_T>
void AllToAll<MPI_T>::alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

template<class MPI_T>
void AllToAll<MPI_T>::alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

#ifdef SWFFT_GPU
template<class MPI_T>
void AllToAll<MPI_T>::alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

template<class MPI_T>
void AllToAll<MPI_T>::alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

#endif

template<class MPI_T>
template<class T>
inline void PairSends<MPI_T>::_alltoall(T* src_buff, T* dest_buff, int n, MPI_Comm comm){
    
    int comm_rank; MPI_Comm_rank(comm,&comm_rank);
    int comm_size; MPI_Comm_size(comm,&comm_size);

    copyBuffers<T> cpy(&dest_buff[comm_rank * n],&src_buff[comm_rank * n],n);
    
    Isend<MPI_T,T> sends[comm_size];
    Irecv<MPI_T,T> recvs[comm_size];

    if (comm_size == 1){
        cpy.wait();
        return;
    }
    if (comm_size == 2){
        this->mpi.sendrecv(&src_buff[((comm_rank + 1)%comm_size) * n],n,(comm_rank + 1)%comm_size,0,&dest_buff[((comm_rank+1)%comm_size) * n],n,(comm_rank + 1)%comm_size,0,comm);
    } else {
        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank)continue;
            sends[i] = this->mpi.isend(&src_buff[i * n],n,i,0,comm);
            recvs[i] = this->mpi.irecv(&dest_buff[i * n],n,i,0,comm);
        }
        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank)continue;
            sends[i].execute();
            recvs[i].execute();
        }
        
        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank)continue;
            sends[i].wait();
            recvs[i].wait();
        }

        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank)continue;
            recvs[i].finalize();
        }
    }

    cpy.wait();

}

template<class MPI_T>
void PairSends<MPI_T>::alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

template<class MPI_T>
void PairSends<MPI_T>::alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

#ifdef SWFFT_GPU
template<class MPI_T>
void PairSends<MPI_T>::alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

template<class MPI_T>
void PairSends<MPI_T>::alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

#endif

template<class MPI_T>
void PairSends<MPI_T>::query(){
    
    printf("CollectiveCommunicator=PairSends\n");

}

template class AllToAll<CPUMPI>;
template class PairSends<CPUMPI>;
template class CollectiveCommunicator<CPUMPI>;

#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI
template class AllToAll<GPUMPI>;
template class PairSends<GPUMPI>;
template class CollectiveCommunicator<GPUMPI>;
#endif
#endif

}
}

#endif