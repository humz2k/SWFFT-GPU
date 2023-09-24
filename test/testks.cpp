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
    int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    n_tests++;
    if(world_rank == 0)printf("Testing %s with T = %s, k_in_blocks = %d and ng = [%d %d %d]\n   ",typeid(SWFFT_T).name(),typeid(T).name(),k_in_blocks,ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,k_in_blocks);
    //printf("my_swfft.buff_sz() = %d\n",my_swfft.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());
    
    bool out;

    //int3 coords = my_swfft.coords();

    //if (world_rank == 0){
    //printf("rank = %d | coords: %d %d %d\n",world_rank,coords.x,coords.y,coords.z);
    //}

    int count = 0;
    for (int i = 0; i < my_swfft.buff_sz(); i++){
        int3 rs = my_swfft.get_rs(i);
        int tmp = rs.x * ngy * ngz + rs.y * ngz + rs.z;
        data[i].x = tmp;
        data[i].y = 0;
    }
    int global,local;
    for (int i = 0; i < 1; i++){

        my_swfft.forward(data,scratch);

        local = 1;
        for (int i = 0; i < my_swfft.buff_sz(); i++){
            int3 ks = my_swfft.get_ks(i);
            int tmp = ks.x * ngy * ngz + ks.y * ngz + ks.z;
            if (((int)data[i].x) != tmp){
                local = 0;
                //printf("rank %d: idx %d = %g not %d\n",world_rank,i,data[i].x,tmp);
                break;
            }
        }
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (world_rank == 0){
            if (global == world_size){
                printf("Passed ks\n   ");
            } else {
                printf("Failed ks...\n   ");
            }
        }

        out = global == world_size;

        my_swfft.backward(data,scratch);

        local = 1;
        for (int i = 0; i < my_swfft.buff_sz(); i++){
            int3 ks = my_swfft.get_rs(i);
            int tmp = ks.x * ngy * ngz + ks.y * ngz + ks.z;
            if (((int)data[i].x) != tmp){
                local = 0;
                break;
            }
        }
        global;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (world_rank == 0){
            if (global == world_size){
                printf("Passed rs\n   ");
            } else {
                printf("Failed rs...\n   ");
            }
        }
        
        out = out && (global == world_size);

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
    #ifdef SWFFT_ALLTOALL
    #ifdef SWFFT_GPU
    test<swfft<AllToAllGPU,CPUMPI,TestFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
    test<swfft<AllToAllGPU,CPUMPI,TestFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
    #endif
    test<swfft<AllToAllCPU,CPUMPI,TestFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
    test<swfft<AllToAllCPU,CPUMPI,TestFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
    #endif
    #ifdef SWFFT_HQFFT
    #ifdef SWFFT_GPU
    test<swfft<HQA2AGPU,CPUMPI,TestFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
    test<swfft<HQA2AGPU,CPUMPI,TestFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
    test<swfft<HQPairGPU,CPUMPI,TestFFT>, complexDoubleHost>(false,ngx,ngy,ngz);
    test<swfft<HQPairGPU,CPUMPI,TestFFT>, complexDoubleHost>(true,ngx,ngy,ngz);
    #endif
    #endif

    //swfft_init_threads(2);

    /*#ifdef SWFFT_HQFFT
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
        //test<swfft<AllToAllGPU,CPUMPI,gpuFFT>, complexDoubleDevice>(false,8);
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
    #endif*/
    
    if(world_rank == 0)printf("%d/%d tests passed\n",n_passed,n_tests);
    MPI_Finalize();
    return 0;
}