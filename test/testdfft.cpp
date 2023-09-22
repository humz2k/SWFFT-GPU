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
template<class SWFFT_T, class T>
bool test(bool k_in_blocks, int ngx, int ngy_ = 0, int ngz_ = 0){
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
    if(world_rank == 0)printf("Testing %s with T = %s, k_in_blocks = %d and ng = [%d %d %d]\n   ",typeid(SWFFT_T).name(),typeid(T).name(),k_in_blocks,ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,k_in_blocks);
    //printf("my_swfft.buff_sz() = %d\n",my_swfft.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());
    
    bool out;

    for (int i = 0; i < 1; i++){

        assign_delta(data,my_swfft.buff_sz());

        my_swfft.forward(data,scratch);

        out = check_kspace(my_swfft,data);

        my_swfft.printLastTime();

        if(world_rank == 0)printf("   ");

        my_swfft.backward(data,scratch);

        out = out && check_rspace(my_swfft,data);

        my_swfft.printLastTime();

        if(world_rank == 0)printf("   ");

    }

    if (out){
        if(world_rank == 0)printf("Passed!\n\n");
        n_passed++;
    } else {
        if(world_rank == 0)printf("Failed...\n\n");
    }
    
    swfftFree(data);
    swfftFree(scratch);

    return out;
    //return false;
}

int main(){
    MPI_Init(NULL,NULL);
    #ifdef SWFFT_GPU
    gpuFree(0);
    #endif
    
    #ifdef SWFFT_PAIRWISE
        #ifdef SWFFT_FFTW
        test<swfft<Pairwise,CPUMPI,fftw>, complexDoubleHost>(false,256);
        test<swfft<Pairwise,CPUMPI,fftw>, complexFloatHost>(false,256);
        #endif
        #ifdef SWFFT_GPU
        #ifdef SWFFT_FFTW
        test<swfft<Pairwise,CPUMPI,fftw>, complexDoubleDevice>(false,256);
        test<swfft<Pairwise,CPUMPI,fftw>, complexFloatDevice>(false,256);
        #endif

        #ifdef SWFFT_CUFFT
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexDoubleHost>(false,256);
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexFloatHost>(false,256);
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexDoubleDevice>(false,256);
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexFloatDevice>(false,256);
        #endif
        #endif
    #endif

    #ifdef SWFFT_ALLTOALL
        #ifdef SWFFT_GPU
        //test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,8);
        #ifdef SWFFT_CUFFT
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatHost>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatHost>(false,256);
        #endif
        #ifdef SWFFT_FFTW
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleDevice>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleDevice>(false,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatDevice>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatDevice>(false,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleHost>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleHost>(false,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatHost>(true,256);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatHost>(false,256);
        #endif
        #ifdef SWFFT_CUFFT
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatHost>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatHost>(false,256);
        #endif
        #ifdef SWFFT_FFTW
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleDevice>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleDevice>(false,256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatDevice>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatDevice>(false,256);
        #endif
        #endif
        #ifdef SWFFT_FFTW
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleHost>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleHost>(false,256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatHost>(true,256);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatHost>(false,256);
        #endif
    #endif

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    if(world_rank == 0)printf("%d/%d tests passed\n",n_passed,n_tests);
    MPI_Finalize();
    return 0;
}