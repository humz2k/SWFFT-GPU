#include "swfft.hpp"
#include "check_kspace.hpp"
#include <stdio.h>
#include <stdlib.h>

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

    #ifdef SWFFT_HQFFT
        #ifdef SWFFT_GPU
            #ifdef SWFFT_CUFFT
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2AGPU,CPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_CUFFT
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<HQA2ACPU,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,CPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
            #endif
        #endif

        #ifdef SWFFT_FFTW
        test<swfft<HQA2ACPU,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<HQA2ACPU,CPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<HQA2ACPU,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
        test<swfft<HQA2ACPU,CPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
        #endif

        #ifdef SWFFT_GPU
            #ifdef SWFFT_CUFFT
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairGPU,CPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_CUFFT
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<HQPairCPU,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,CPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
            #endif
        #endif

        #ifdef SWFFT_FFTW
        test<swfft<HQPairCPU,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<HQPairCPU,CPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<HQPairCPU,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
        test<swfft<HQPairCPU,CPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
        #endif
    #endif

    #ifdef SWFFT_PAIRWISE
        #ifdef SWFFT_FFTW
        test<swfft<Pairwise,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<Pairwise,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
        #endif
        #ifdef SWFFT_GPU
        #ifdef SWFFT_FFTW
        test<swfft<Pairwise,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<Pairwise,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
        #endif

        #ifdef SWFFT_CUFFT
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<Pairwise,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
        #endif
        #endif
    #endif
    
    #ifdef SWFFT_ALLTOALL
        #ifdef SWFFT_GPU
        #ifdef SWFFT_CUFFT
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
        #endif
        #ifdef SWFFT_FFTW
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllGPU,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
        #endif
        #ifdef SWFFT_CUFFT
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
        #endif
        #ifdef SWFFT_FFTW
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
        #endif
        #endif
        #ifdef SWFFT_FFTW
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
        test<swfft<AllToAllCPU,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
        #endif
    #endif

    #ifdef SWFFT_GPU
    #ifdef SWFFT_CUFFT
    #ifdef SWFFT_GPUDELEGATE
    test<swfft<GPUDelegate,CPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
    test<swfft<GPUDelegate,CPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
    test<swfft<GPUDelegate,CPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
    test<swfft<GPUDelegate,CPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
    #endif
    #endif
    #endif

    #ifdef SWFFT_GPU
    #ifndef SWFFT_NOCUDAMPI
        #ifdef SWFFT_HQFFT
            #ifdef SWFFT_GPU
                #ifdef SWFFT_CUFFT
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
                #endif
                #ifdef SWFFT_FFTW
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
                test<swfft<HQA2AGPU,GPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
                #endif
                #ifdef SWFFT_CUFFT
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
                #endif
                #ifdef SWFFT_FFTW
                test<swfft<HQA2ACPU,GPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQA2ACPU,GPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
                #endif
            #endif

            #ifdef SWFFT_FFTW
            test<swfft<HQA2ACPU,GPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,GPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,GPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQA2ACPU,GPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif

            #ifdef SWFFT_GPU
                #ifdef SWFFT_CUFFT
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
                #endif
                #ifdef SWFFT_FFTW
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
                test<swfft<HQPairGPU,GPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
                #endif
                #ifdef SWFFT_CUFFT
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
                #endif
                #ifdef SWFFT_FFTW
                test<swfft<HQPairCPU,GPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
                test<swfft<HQPairCPU,GPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
                #endif
            #endif

            #ifdef SWFFT_FFTW
            test<swfft<HQPairCPU,GPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,GPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<HQPairCPU,GPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
            test<swfft<HQPairCPU,GPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
            #endif
        #endif
        
        #ifdef SWFFT_ALLTOALL
            #ifdef SWFFT_GPU
            #ifdef SWFFT_CUFFT
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllGPU,GPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_CUFFT
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexDoubleDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexFloatDevice>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
            #endif
            #endif
            #ifdef SWFFT_FFTW
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexDoubleHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexFloatHost>(true,ngx,ngy,ngz);
            test<swfft<AllToAllCPU,GPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
            #endif
        #endif

        #ifdef SWFFT_GPU
        #ifdef SWFFT_CUFFT
        #ifdef SWFFT_GPUDELEGATE
        test<swfft<GPUDelegate,GPUMPI,gpuFFT>, complexDoubleDevice>(true,ngx,ngy,ngz);
        test<swfft<GPUDelegate,GPUMPI,gpuFFT>, complexFloatDevice>(true,ngx,ngy,ngz);
        test<swfft<GPUDelegate,GPUMPI,gpuFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
        test<swfft<GPUDelegate,GPUMPI,gpuFFT>, complexFloatHost>(true,ngx,ngy,ngz);
        #endif
        #endif
        #endif

        #ifdef SWFFT_SMARTMAP
        #ifdef SWFFT_FFTW
        test<swfft<SmartFFT,CPUMPI,fftw>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<SmartFFT,CPUMPI,fftw>, complexFloatHost>(false,ngx,ngy,ngz);
        #ifdef SWFFT_GPU
        test<swfft<SmartFFT,CPUMPI,fftw>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<SmartFFT,CPUMPI,fftw>, complexFloatDevice>(false,ngx,ngy,ngz);
        #endif
        #endif
        #ifdef SWFFT_GPU
        #ifdef SWFFT_CUFFT
        test<swfft<SmartFFT,CPUMPI,gpuFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
        test<swfft<SmartFFT,CPUMPI,gpuFFT>, complexFloatHost>(false,ngx,ngy,ngz);
        test<swfft<SmartFFT,CPUMPI,gpuFFT>, complexDoubleDevice>(false,ngx,ngy,ngz);
        test<swfft<SmartFFT,CPUMPI,gpuFFT>, complexFloatDevice>(false,ngx,ngy,ngz);
        #endif
        #endif
        #endif
    #endif
    #endif


    if(world_rank == 0){
        printf("%d/%d tests passed\n",n_passed,n_tests);
    }
    MPI_Finalize();
    return 0;
}