#include "swfft.hpp"
#include "check_kspace.hpp"
#include <stdio.h>
#include <stdlib.h>

int n_tests = 0;
int n_passed = 0;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define BLOCKSIZE 64

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

#ifdef GPU
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
template<class SWFFT_T, class T>
bool test(int ngx, int ngy_ = 0, int ngz_ = 0){
    int ngy = ngy_;
    int ngz = ngz_;
    if (ngy == 0){
        ngy = ngx;
    }
    if (ngz == 0){
        ngz = ngx;
    }
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    n_tests++;
    if(world_rank == 0)printf("Testing %s with T = %s and ng = [%d %d %d]\n   ",typeid(SWFFT_T).name(),typeid(T).name(),ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,true);

    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());

    assign_delta(data,my_swfft.buff_sz());

    my_swfft.forward(data,scratch);

    bool out = check_kspace(my_swfft,data);

    my_swfft.backward(data,scratch);

    out = out && check_rspace(my_swfft,data);

    if (out){
        if(world_rank == 0)printf("Passed!\n\n");
        n_passed++;
    }

    swfftFree(data);
    swfftFree(scratch);

    return out;
}

int main(){
    MPI_Init(NULL,NULL);
    #ifdef ALLTOALL
        #ifdef GPU
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatDevice>(256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleHost>(256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatHost>(256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleDevice>(256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatDevice>(256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleHost>(256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatHost>(256);

        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatDevice>(256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleHost>(256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatHost>(256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleDevice>(256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatDevice>(256);
        #endif
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleHost>(256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatHost>(256);
    #endif

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    if(world_rank == 0)printf("%d/%d tests passed\n",n_passed,n_tests);
    MPI_Finalize();
    return 0;
}