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
bool test(bool blocks, MPI_Comm comm, int ngx, int ngy_ = 0, int ngz_ = 0){
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
    if(world_rank == 0)printf("   Testing %s with k_in_blocks = %d and ng = [%d %d %d]\n   ",typeid(SWFFT_T).name(),blocks,ngx,ngy,ngz);
    SWFFT_T my_swfft(comm,ngx,ngy,ngz,BLOCKSIZE,blocks);

    if (my_swfft.test_distribution()){
        if(world_rank == 0)printf("   Passed!\n\n");
        n_passed++;
        return true;
    }
    if(world_rank == 0)printf("   Failed...\n\n");
    return false;
}

int main(){
    MPI_Init(NULL,NULL);
    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size;MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    #ifdef ALLTOALL
        if (world_size >= 8){
            if(world_rank == 0)printf("Testing with world_size = 8:\n\n");
            MPI_Comm comm;
            MPI_Comm_split(MPI_COMM_WORLD,world_rank < 8,world_rank,&comm);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16,8,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16,8,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,8,16,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,8,16,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,8,16,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,8,16,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,8,8,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,8,8,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16,8,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16,8,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16,16,8);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16,16,8);
            MPI_Comm_free(&comm);
        }
        if (world_size >= 16){
            if(world_rank == 0)printf("Testing with world_size = 16:\n\n");
            MPI_Comm comm;
            MPI_Comm_split(MPI_COMM_WORLD,world_rank < 16,world_rank,&comm);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,32,16,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,32,16,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16,32,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16,32,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16,32,32);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16,32,32);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,16,16,32);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,16,16,32);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,32,16,32);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,32,16,32);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(true,comm,32,32,16);
            test<swfft<AllToAllGPU,CPUMPI,gpuFFT>>(false,comm,32,32,16);
            MPI_Comm_free(&comm);
        }
    #endif

    if(world_rank == 0)printf("%d/%d tests passed\n",n_passed,n_tests);
    MPI_Finalize();
    return 0;
}