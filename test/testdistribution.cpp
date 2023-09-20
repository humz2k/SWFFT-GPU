#include "swfft.hpp"
#include "check_kspace.hpp"
#include <stdio.h>
#include <stdlib.h>

int n_tests = 0;
int n_passed = 0;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define BLOCKSIZE 64

template<class SWFFT_T>
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
    if(world_rank == 0)printf("Testing %s and ng = [%d %d %d]\n   ",typeid(SWFFT_T).name(),ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,true);

    if (my_swfft.test_distribution()){
        if(world_rank == 0)printf("Passed!\n\n");
        n_passed++;
        return true;
    }
    if(world_rank == 0)printf("Failed...\n\n");
    return false;
}

int main(){
    MPI_Init(NULL,NULL);
    #ifdef ALLTOALL
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(10);
        //test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(256);
    #endif

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    if(world_rank == 0)printf("%d/%d tests passed\n",n_passed,n_tests);
    MPI_Finalize();
    return 0;
}