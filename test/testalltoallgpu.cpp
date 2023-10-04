#include "swfft.hpp"
#include "check_kspace.hpp"
#include <stdio.h>
#include <stdlib.h>

#if defined(SWFFT_GPU) && defined(SWFFT_CUFFT) && defined(SWFFT_ALLTOALL)

using namespace SWFFT;

int n_tests = 0;
int n_passed = 0;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define BLOCKSIZE 64

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
    if(world_rank == 0)printf("Testing %s with T = %s, k_in_blocks = %d and ng = [%d %d %d]\n",typeid(SWFFT_T).name(),typeid(T).name(),k_in_blocks,ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,k_in_blocks);
    my_swfft.query();
    if(world_rank == 0)printf("   ");
    //printf("my_swfft.buff_sz() = %d\n",my_swfft.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());
    
    bool out = false;

    for (int i = 0; i < 1; i++){

        assign_delta(data,my_swfft.buff_sz());

        my_swfft.forward(data,scratch);
        my_swfft.synchronize();

        out = check_kspace(my_swfft,data);

        my_swfft.printLastTime();

        if(world_rank == 0)printf("   ");

        my_swfft.backward(data,scratch);
        my_swfft.synchronize();

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

int main(int argc, char** argv){
    MPI_Init(NULL,NULL);
    #ifdef SWFFT_GPU
    gpuFree(0);
    #endif

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    if (!((argc == 2) || (argc == 4))){
        if(world_rank == 0)printf("USAGE: %s <ngx> [ngy ngz]\n", argv[0]);
        MPI_Finalize();
        return -1;
    }
    
    int ngx = atoi(argv[1]);
    int ngy = ngx;
    int ngz = ngx;
    if (argc == 4){
        ngy = atoi(argv[2]);
        ngz = atoi(argv[3]);
    }

    //swfft_init_threads(2);

    test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
    
    if(world_rank == 0)printf("%d/%d tests passed\n",n_passed,n_tests);
    MPI_Finalize();
    return 0;
}

#else

int main(){
    MPI_Init(NULL,NULL);

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    if(world_rank == 0){
        printf("Not compiled with AllToAllGPU or cuFFT!\n");
    }

    MPI_Finalize();
}

#endif