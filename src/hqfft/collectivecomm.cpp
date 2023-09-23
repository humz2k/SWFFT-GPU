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

template<class T>
class copyBuffers{
    private:
        T* dest;
        T* src;
        int n;
        #ifdef SWFFT_GPU
        gpuEvent_t event;
        #endif
        
    public:
        copyBuffers(T* dest_, T* src_, int n_);
        ~copyBuffers();
        void wait();
};

template<class T>
copyBuffers<T>::copyBuffers(T* dest_, T* src_, int n_) : dest(dest_), src(src_), n(n_){
    for (int i = 0; i < n; i++){
        dest[i] = src[i];
    }
}

template<class T>
copyBuffers<T>::~copyBuffers(){

}

template<class T>
void copyBuffers<T>::wait(){

}
#ifdef SWFFT_GPU

template<>
copyBuffers<complexDoubleDevice>::~copyBuffers(){

}

template<>
copyBuffers<complexFloatDevice>::~copyBuffers(){

}

template<>
copyBuffers<complexDoubleDevice>::copyBuffers(complexDoubleDevice* dest_, complexDoubleDevice* src_, int n_) : dest(dest_), src(src_), n(n_){
    gpuEventCreate(&event);
    gpuMemcpyAsync(dest,src,n * sizeof(complexDoubleDevice),gpuMemcpyDeviceToDevice);
    gpuEventRecord(event);
}

template<>
void copyBuffers<complexDoubleDevice>::wait(){
    gpuEventSynchronize(event);
    gpuEventDestroy(event);
}

template<>
copyBuffers<complexFloatDevice>::copyBuffers(complexFloatDevice* dest_, complexFloatDevice* src_, int n_) : dest(dest_), src(src_), n(n_){
    gpuEventCreate(&event);
    gpuMemcpyAsync(dest,src,n * sizeof(complexFloatDevice),gpuMemcpyDeviceToDevice);
    gpuEventRecord(event);
}

template<>
void copyBuffers<complexFloatDevice>::wait(){
    gpuEventSynchronize(event);
    gpuEventDestroy(event);
}
#endif

template<>
template<class T>
inline void PairSends<CPUMPI>::_alltoall(T* src_buff, T* dest_buff, int n, MPI_Comm comm){
    
    int comm_rank; MPI_Comm_rank(comm,&comm_rank);
    int comm_size; MPI_Comm_size(comm,&comm_size);

    copyBuffers<T> cpy(&dest_buff[comm_rank * n],&src_buff[comm_rank * n],n);
    
    CPUIsend<T> sends[comm_size];
    CPUIrecv<T> recvs[comm_size];

    if (comm_size == 2){
        this->mpi.sendrecv(&src_buff[((comm_rank + 1)%comm_size) * n],n,(comm_rank + 1)%comm_size,0,&dest_buff[((comm_rank+1)%comm_size) * n],n,(comm_rank + 1)%comm_size,0,comm);
    } else {
        for (int i = 0; i < comm_size; i++){
            if (i == comm_rank)continue;
            CPUIsend<T> this_send = this->mpi.isend(&src_buff[i * n],n,i,0,comm);
            CPUIrecv<T> this_recv = this->mpi.irecv(&dest_buff[i * n],n,i,0,comm);
            sends[i] = this_send;
            recvs[i] = this_recv;
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

template<>
void PairSends<CPUMPI>::alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

template<>
void PairSends<CPUMPI>::alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

#ifdef SWFFT_GPU
template<>
void PairSends<CPUMPI>::alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

template<>
void PairSends<CPUMPI>::alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm){
    
    _alltoall(src,dest,n_recv,comm);    

}

#endif

template<>
void PairSends<CPUMPI>::query(){
    
    printf("CollectiveCommunicator=PairSends\n");

}

template class AllToAll<CPUMPI>;
template class PairSends<CPUMPI>;

}
}

#endif