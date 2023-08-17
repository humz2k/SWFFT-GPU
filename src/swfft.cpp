#include "swfft.hpp"

template<class Backend, class T>
swfft<Backend,T>::swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,comm){}

template<class Backend, class T>
swfft<Backend,T>::swfft(int ngx, int ngy, int ngz, MPI_Comm comm) : backend(ngx,ngy,ngz,64,comm){}

template<class Backend, class T>
swfft<Backend,T>::swfft(int ng, MPI_Comm comm) : backend(ng,ng,ng,64,comm){}

template<class Backend, class T>
swfft<Backend,T>::swfft(int ng, int blockSize, MPI_Comm comm) : backend(ng,ng,ng,blockSize,comm){}

template<class Backend, class T>
swfft<Backend,T>::~swfft(){}

template<class Backend, class T>
void swfft<Backend,T>::makePlans(){
    backend.makePlans();
}

template<class Backend, class T>
void swfft<Backend,T>::makePlans(T* buff2){
    backend.makePlans(buff2);
}

template<class Backend, class T>
void swfft<Backend,T>::makePlans(T* buff1, T* buff2){
    backend.makePlans(buff1, buff2);
}

template<class Backend, class T>
void swfft<Backend,T>::forward(){
    backend.forward();
}

template<class Backend, class T>
void swfft<Backend,T>::forward(T* buff1){
    backend.forward(buff1);
}

template<class Backend, class T>
void swfft<Backend,T>::backward(){
    backend.backward();
}

template<class Backend, class T>
void swfft<Backend,T>::backward(T* buff1){
    backend.backward(buff1);
}