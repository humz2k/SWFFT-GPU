#include "swfft.hpp"
#define CHECK_SILENT
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
    //my_swfft.query();
    if(world_rank == 0)printf("   ");
    //printf("my_swfft.buff_sz() = %d\n",my_swfft.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());
    
    bool out = false;

    timing_stats_t initial_f;
    timing_stats_t initial_b;
    timing_stats_t avg_f = {};
    timing_stats_t avg_b = {};

    int avg_of = 10;

    for (int i = 0; i < (avg_of + 1); i++){

        assign_delta(data,my_swfft.buff_sz());

        my_swfft.forward(data,scratch);
        my_swfft.synchronize();

        out = check_kspace(my_swfft,data);

        timing_stats_t forward = my_swfft.printLastTime();

        if(world_rank == 0)printf("   ");

        my_swfft.backward(data,scratch);
        my_swfft.synchronize();

        out = out && check_rspace(my_swfft,data);

        timing_stats_t backward = my_swfft.printLastTime();

        if(world_rank == 0)printf("   ");

        if (i == 0){
            initial_f = forward;
            initial_b = backward;
        } else {
            avg_f.avg += forward.avg;
            avg_f.max += forward.max;
            avg_f.min += forward.min;

            avg_b.avg += backward.avg;
            avg_b.max += backward.max;
            avg_b.min += backward.min;
        }

    }

    if (world_rank == 0){
        printf("INITIAL TIMES:\n");
        printf("   Forward: avg = %g, max = %g, min = %g\n",initial_f.avg,initial_f.max,initial_f.min);
        printf("   Backward: avg = %g, max = %g, min = %g\n",initial_b.avg,initial_b.max,initial_b.min);
        printf("AVG OF %d FFTS:\n",avg_of);
        printf("   Forward: avg = %g, max = %g, min = %g\n",avg_f.avg / ((double)avg_of),avg_f.max / ((double)avg_of),avg_f.min / ((double)avg_of));
        printf("   Backward: avg = %g, max = %g, min = %g\n",avg_b.avg / ((double)avg_of),avg_b.max / ((double)avg_of),avg_b.min / ((double)avg_of));
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

template<class T, class FFTBackend>
void benchmark_cpu(bool k_in_blocks, int ngx, int ngy, int ngz){

    #ifdef SWFFT_HQFFT
    test<swfft<HQA2ACPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    test<swfft<HQPairCPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    #endif
    #ifdef SWFFT_ALLTOALL
    test<swfft<AllToAllCPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    #endif
    #ifdef SWFFT_PAIRWISE
    if (!k_in_blocks){
        test<swfft<Pairwise,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    }
    #endif

}

template<class T, class FFTBackend>
void benchmark_gpu(bool k_in_blocks, int ngx, int ngy, int ngz){
    #ifdef SWFFT_GPU
    #ifdef SWFFT_HQFFT
    test<swfft<HQA2AGPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    test<swfft<HQPairGPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    #endif
    #ifdef SWFFT_ALLTOALL
    test<swfft<AllToAllGPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    #endif
    #ifdef SWFFT_GPUDELEGATE
    //if(k_in_blocks){
    test<swfft<GPUDelegate,OPTMPI,gpuFFT>,T>(true,ngx,ngy,ngz);
    //}
    #endif    
    #endif
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
    #ifdef SWFFT_FFTW
    benchmark_cpu<complexDoubleHost,fftw>(false,ngx,ngy,ngz);
    #endif

    #if defined(SWFFT_GPU) && defined(SWFFT_CUFFT)
    benchmark_gpu<complexDoubleDevice,gpuFFT>(false,ngx,ngy,ngz);
    #endif
    
    if(world_rank == 0){
        printf("%d/%d tests passed\n",n_passed,n_tests);
    }
    MPI_Finalize();
    return 0;
}