#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "alltoall.hpp"

template<class T, class FFTBackend>
class SwfftBackend{
    public:
        SwfftBackend();
        SwfftBackend(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        SwfftBackend(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        SwfftBackend(int ngx, int ngy, int ngz, MPI_Comm comm);
        SwfftBackend(int ng, int blockSize, MPI_Comm comm);
        SwfftBackend(int ng, MPI_Comm comm);

        ~SwfftBackend();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

template<class T, class FFTBackend>
class AllToAll : public SwfftBackend<T, FFTBackend>{
    public:
        A2A::Distribution<T> dist;
        A2A::Dfft<T,FFTBackend> dfft;

        AllToAll(){};
        AllToAll(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm) : dist(comm,ngx,blockSize,batches), dfft(dist){};
        AllToAll(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm) : dist(comm,ngx,blockSize,1), dfft(dist){};
        AllToAll(int ngx, int ngy, int ngz, MPI_Comm comm) : dist(comm,ngx,64,1), dfft(dist){};
        AllToAll(int ng, int blockSize, MPI_Comm comm) : dist(comm,ng,blockSize,1), dfft(dist){};
        AllToAll(int ng, MPI_Comm comm) : dist(comm,ng,64,1), dfft(dist){};

        ~AllToAll(){};

        void makePlans(T* buff1, T* buff2) {dfft.makePlans(buff1,buff2);};
        void makePlans(T* buff2) {dfft.makePlans(buff2);};
        void makePlans() {dfft.makePlans();};

        void forward() {dfft.forward();};
        void forward(T* buff1) {dfft.forward(buff1);};
        void backward() {dfft.backward()};
        void backward(T* buff1) {dfft.backward(buff1)};
};

template<class T, class FFTBackend>
class P3Collective : public SwfftBackend<T,FFTBackend>{
    public:
        P3Collective();
        P3Collective(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        P3Collective(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        P3Collective(int ngx, int ngy, int ngz, MPI_Comm comm);
        P3Collective(int ng, int blockSize, MPI_Comm comm);
        P3Collective(int ng, MPI_Comm comm);

        ~P3Collective();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

template<class T, class FFTBackend>
class P3SendRecv : public SwfftBackend<T,FFTBackend>{
    public:
        P3SendRecv();
        P3SendRecv(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        P3SendRecv(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        P3SendRecv(int ngx, int ngy, int ngz, MPI_Comm comm);
        P3SendRecv(int ng, int blockSize, MPI_Comm comm);
        P3SendRecv(int ng, MPI_Comm comm);

        ~P3SendRecv();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

template<class T, class FFTBackend>
class Pairwise : public SwfftBackend<T,FFTBackend>{
    public:
        Pairwise();
        Pairwise(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm);
        Pairwise(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        Pairwise(int ngx, int ngy, int ngz, MPI_Comm comm);
        Pairwise(int ng, int blockSize, MPI_Comm comm);
        Pairwise(int ng, MPI_Comm comm);

        ~Pairwise();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

template<template<class,class> class Backend, class T, class FFTBackend>
class swfft{
    private:
        Backend<T,FFTBackend> backend;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, int batches, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,batches,comm){};
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm) : backend(ngx,ngy,ngz,blockSize,comm){};
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm) : backend(ngx,ngy,ngz,comm){};
        swfft(int ng, int blockSize, MPI_Comm comm) : backend(ng,blockSize,comm){};
        swfft(int ng, MPI_Comm comm) : backend(ng,comm){};
        
        ~swfft();

        void makePlans(T* buff1, T* buff2){
            backend.makePlans(buff1,buff2);
        };
        void makePlans(T* buff2){
            backend.makePlans(buff2);
        };
        void makePlans(){
            backend.makePlans();
        };

        void forward(){
            backend.forward();
        };
        void forward(T* buff1){
            backend.forward(buff1);
        };
        void backward(){
            backend.backward();
        };
        void backward(T* buff1){
            backend.backward(buff1);
        };

};