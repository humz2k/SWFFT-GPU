#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

template<class T>
class SwfftBackend{
    public:
        SwfftBackend();
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

template<class T>
class AllToAll : public SwfftBackend<T>{
    public:
        AllToAll();
        AllToAll(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        AllToAll(int ngx, int ngy, int ngz, MPI_Comm comm);
        AllToAll(int ng, int blockSize, MPI_Comm comm);
        AllToAll(int ng, MPI_Comm comm);

        ~AllToAll();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

template<template <class> class Dist, class T>
class P3 : public SwfftBackend<T>{
    public:
        P3();
        P3(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        P3(int ngx, int ngy, int ngz, MPI_Comm comm);
        P3(int ng, int blockSize, MPI_Comm comm);
        P3(int ng, MPI_Comm comm);

        ~P3();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);
};

template<class T>
class Pairwise : public SwfftBackend<T>{
    public:
        Pairwise();
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

template<template<class> class Backend, class T>
class swfft{
    private:
        Backend<T> backend;
        int ngx;
        int ngy;
        int ngz;
        int blockSize;
        MPI_Comm comm;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm);
        swfft(int ng, int blockSize, MPI_Comm comm);
        swfft(int ng, MPI_Comm comm);
        
        ~swfft();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);

};

template<template<class,class> class Backend, class Dist, class T>
class swfft{
    private:
        Backend<Dist,T> backend;
        int ngx;
        int ngy;
        int ngz;
        int blockSize;
        MPI_Comm comm;

    public:
        swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);
        swfft(int ngx, int ngy, int ngz, MPI_Comm comm);
        swfft(int ng, int blockSize, MPI_Comm comm);
        swfft(int ng, MPI_Comm comm);
        
        ~swfft();

        void makePlans(T* buff1, T* buff2);
        void makePlans(T* buff2);
        void makePlans();

        void forward();
        void forward(T* buff1);
        void backward();
        void backward(T* buff1);

};