
#ifndef SWFFT_ALLTOALL
int main(){
    return 0;
}
#else
#include "alltoall.hpp"
#include "check_kspace.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

using namespace SWFFT;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define BLOCKSIZE 64

int n_tests = 0;
int n_passed = 0;

#define IS_TRUE(func,T,MPI_T,ngx,ngy,ngz) { n_tests += 1; int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); if(world_rank == 0){std::cout << "Test:\n   AllToAllGPU\n   " << TOSTRING(func) << "\n   " << TOSTRING(T) << "\n   " << TOSTRING(MPI_T) << "\n   Ngx=" << TOSTRING(ngx) << ",Ngx=" << TOSTRING(ngy) << ",Ngz=" << TOSTRING(ngz) << "\n   ";} if (!(test<func,T,MPI_T>(ngx,ngy,ngz))){ if(world_rank == 0){std::cout << "failed on line " << __LINE__ << "\n" << std::endl;}}else{if(world_rank == 0){std::cout  << "passed\n" << std::endl; n_passed += 1;}} }

#define IS_TRUECPU(func,T,MPI_T,ngx,ngy,ngz) { n_tests += 1; int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); if(world_rank == 0){std::cout << "Test:\n   AllToAllCPU\n   " << TOSTRING(func) << "\n   " << TOSTRING(T) << "\n   " << TOSTRING(MPI_T) << "\n   Ngx=" << TOSTRING(ngx) << ",Ngx=" << TOSTRING(ngy) << ",Ngz=" << TOSTRING(ngz) << "\n   ";} if (!(testcpu<func,T,MPI_T>(ngx,ngy,ngz))){ if(world_rank == 0){std::cout << "failed on line " << __LINE__ << "\n" << std::endl;}}else{if(world_rank == 0){std::cout  << "passed\n" << std::endl; n_passed += 1;}} }

void assign_delta(complexDoubleHost* data, int NG){
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    for (int i = 0; i < NG; i++){
        data[i].x = 0;
        data[i].y = 0;
    }
    if (world_rank == 0){
        data[0].x = 1;
        data[0].y = 0;
    }
}

void assign_delta(complexFloatHost* data, int NG){
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    for (int i = 0; i < NG; i++){
        data[i].x = 0;
        data[i].y = 0;
    }
    if (world_rank == 0){
        data[0].x = 1;
        data[0].y = 0;
    }
}

#ifdef SWFFT_GPU
void assign_delta(complexDoubleDevice* data, int NG){
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    gpuMemset(data,0,sizeof(complexDoubleDevice)*NG);
    if (world_rank == 0){
        complexDoubleDevice start;
        start.x = 1;
        start.y = 0;
        gpuMemcpy(data,&start,sizeof(complexDoubleDevice),cudaMemcpyHostToDevice);
    }
}

void assign_delta(complexFloatDevice* data, int NG){
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    gpuMemset(data,0,sizeof(complexFloatDevice)*NG);
    if (world_rank == 0){
        complexFloatDevice start;
        start.x = 1;
        start.y = 0;
        gpuMemcpy(data,&start,sizeof(complexFloatDevice),cudaMemcpyHostToDevice);
    }
}
#endif

#ifdef SWFFT_GPU
template<class MPI_T, class FFTBackend, class T>
bool test(int ngx, int ngy_=0, int ngz_=0){
    int ngy = ngy_;
    int ngz = ngz_;
    if ((ngy_ = 0) && (ngz_ == 0)){
        ngy = ngx;
        ngz = ngx;
    }
    AllToAllGPU<MPI_T,FFTBackend> alltoall(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE);
    //printf("BUFF_SZ = %d\n",alltoall.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * alltoall.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * alltoall.buff_sz());
    //printf("BUFF_SZ = %d\n",alltoall.buff_sz());


    assign_delta(data,alltoall.buff_sz());

    alltoall.forward(data,scratch);

    bool out = check_kspace(alltoall,data);

    swfftFree(data);
    swfftFree(scratch);
    return out;

}
#endif

template<class MPI_T, class FFTBackend, class T>
bool testcpu(int ngx, int ngy_=0, int ngz_=0){
    int ngy = ngy_;
    int ngz = ngz_;
    if ((ngy_ = 0) && (ngz_ == 0)){
        ngy = ngx;
        ngz = ngx;
    }
    AllToAllCPU<MPI_T,FFTBackend> alltoall(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE);
    //printf("BUFF_SZ = %d\n",alltoall.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * alltoall.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * alltoall.buff_sz());
    //printf("BUFF_SZ = %d\n",alltoall.buff_sz());


    assign_delta(data,alltoall.buff_sz());

    alltoall.forward(data,scratch);

    bool out = check_kspace(alltoall,data);

    swfftFree(data);
    swfftFree(scratch);
    return out;

}

int main(){
    MPI_Init(NULL,NULL);
    
    #ifdef SWFFT_GPU
    #ifdef SWFFT_CUFFT
    IS_TRUE(CPUMPI,gpuFFT,complexDoubleDevice,256,256,256);
    IS_TRUE(CPUMPI,gpuFFT,complexFloatDevice,256,256,256);
    IS_TRUE(CPUMPI,gpuFFT,complexDoubleHost,256,256,256);
    IS_TRUE(CPUMPI,gpuFFT,complexFloatHost,256,256,256);
    #endif

    #ifdef SWFFT_FFTW
    IS_TRUE(CPUMPI,fftw,complexDoubleDevice,256,256,256);
    IS_TRUE(CPUMPI,fftw,complexFloatDevice,256,256,256);
    IS_TRUE(CPUMPI,fftw,complexDoubleHost,256,256,256);
    IS_TRUE(CPUMPI,fftw,complexFloatHost,256,256,256);
    #endif

    #ifdef SWFFT_CUFFT
    IS_TRUECPU(CPUMPI,gpuFFT,complexDoubleDevice,256,256,256);
    IS_TRUECPU(CPUMPI,gpuFFT,complexFloatDevice,256,256,256);
    IS_TRUECPU(CPUMPI,gpuFFT,complexDoubleHost,256,256,256);
    IS_TRUECPU(CPUMPI,gpuFFT,complexFloatHost,256,256,256);
    #endif

    #ifdef SWFFT_FFTW
    IS_TRUECPU(CPUMPI,fftw,complexDoubleDevice,256,256,256);
    IS_TRUECPU(CPUMPI,fftw,complexFloatDevice,256,256,256);
    #endif
    #endif
    #ifdef SWFFT_FFTW
    IS_TRUECPU(CPUMPI,fftw,complexDoubleHost,256,256,256);
    IS_TRUECPU(CPUMPI,fftw,complexFloatHost,256,256,256);
    #endif

    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    if(world_rank == 0){
        printf("%d/%d tests passed\n",n_passed,n_tests);
    }
    MPI_Finalize();
    return 0;
}
#endif