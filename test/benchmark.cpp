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
bool test(bool k_in_blocks, int ngx, int ngy, int ngz, int nreps = 10){
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    n_tests++;
    if(world_rank == 0)printf("Benchmarking %s with T = %s, k_in_blocks = %d and ng = [%d %d %d]\n   ",typeid(SWFFT_T).name(),typeid(T).name(),k_in_blocks,ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,k_in_blocks);
    my_swfft.query();
    //printf("my_swfft.buff_sz() = %d\n",my_swfft.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());
    
    bool out;

    for (int i = 0; i < nreps; i++){

        if(world_rank == 0)printf("\nTest %d\n",i);

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

void run_benckmark(char** argv, int world_rank, int ngx, int ngy, int ngz, bool k_in_blocks, int nreps){
    if (!strcmp(argv[1],"Pairwise")){
        #ifndef SWFFT_PAIRWISE
        if(world_rank == 0)printf("Not compiled with Pairwise!\n");
        return;
        #else
        if (k_in_blocks){
            if(world_rank == 0)printf("Can't run Pairwise with k_in_blocks=true\n");
            return;
        }
        if (!strcmp(argv[2],"CPUMPI")){

            if (!strcmp(argv[3],"fftw")){
                #ifndef SWFFT_FFTW
                if(world_rank == 0)printf("Not compiled with fftw!\n");
                return;
                #else

                if (!strcmp(argv[4],"complexDoubleHost")){
                    test<swfft<Pairwise,CPUMPI,fftw>,complexDoubleHost>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatHost")){
                    test<swfft<Pairwise,CPUMPI,fftw>,complexFloatHost>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexDoubleDevice")){
                    #ifndef SWFFT_GPU
                    if(world_rank == 0)printf("Not compiled with USE_GPU=true!\n");
                    #else
                    test<swfft<Pairwise,CPUMPI,fftw>,complexDoubleDevice>(false,ngx,ngy,ngz,nreps);
                    #endif
                    return;
                }
                if (!strcmp(argv[4],"complexFloatDevice")){
                    #ifndef SWFFT_GPU
                    if(world_rank == 0)printf("Not compiled with USE_GPU=true!\n");
                    #else
                    test<swfft<Pairwise,CPUMPI,fftw>,complexFloatDevice>(false,ngx,ngy,ngz,nreps);
                    #endif
                    return;
                }

                #endif
            }

            if(!strcmp(argv[3],"gpuFFT")){
                #ifndef SWFFT_GPU
                if(world_rank == 0)printf("Not compiled with SWFFT_GPU!\n");
                return;
                #else
                #ifndef SWFFT_CUFFT
                if(world_rank == 0)printf("Not compiled with cuFFT!\n");
                return;
                #else
                if (!strcmp(argv[4],"complexDoubleHost")){
                    test<swfft<Pairwise,CPUMPI,gpuFFT>,complexDoubleHost>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatHost")){
                    test<swfft<Pairwise,CPUMPI,gpuFFT>,complexFloatHost>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexDoubleDevice")){
                    test<swfft<Pairwise,CPUMPI,gpuFFT>,complexDoubleDevice>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatDevice")){
                    test<swfft<Pairwise,CPUMPI,gpuFFT>,complexFloatDevice>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                #endif
                #endif
            }

        } else {
            if(world_rank == 0)printf("Invalid mpi_t (%s)\n",argv[2]);
            return;
        }
        #endif
    }

    if (!strcmp(argv[1],"AllToAllGPU")){
        #ifndef SWFFT_GPU
        if(world_rank == 0)printf("Not compiled with SWFFT_GPU\n");
            return;
        #else
        #ifndef SWFFT_ALLTOALL
        if(world_rank == 0)printf("Not compiled with AllToAll!\n");
        return;
        #else
        if (!strcmp(argv[2],"CPUMPI")){

            if (!strcmp(argv[3],"fftw")){
                #ifndef SWFFT_FFTW
                if(world_rank == 0)printf("Not compiled with fftw!\n");
                return;
                #else

                if (!strcmp(argv[4],"complexDoubleHost")){
                    test<swfft<AllToAllGPU,CPUMPI,fftw>,complexDoubleHost>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatHost")){
                    test<swfft<AllToAllGPU,CPUMPI,fftw>,complexFloatHost>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexDoubleDevice")){
                    test<swfft<AllToAllGPU,CPUMPI,fftw>,complexDoubleDevice>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatDevice")){
                    test<swfft<AllToAllGPU,CPUMPI,fftw>,complexFloatDevice>(false,ngx,ngy,ngz,nreps);
                    return;
                }

                #endif
            }

            if(!strcmp(argv[3],"gpuFFT")){
                #ifndef SWFFT_CUFFT
                if(world_rank == 0)printf("Not compiled with cuFFT!\n");
                return;
                #else
                if (!strcmp(argv[4],"complexDoubleHost")){
                    test<swfft<AllToAllGPU,CPUMPI,gpuFFT>,complexDoubleHost>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatHost")){
                    test<swfft<AllToAllGPU,CPUMPI,gpuFFT>,complexFloatHost>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexDoubleDevice")){
                    test<swfft<AllToAllGPU,CPUMPI,gpuFFT>,complexDoubleDevice>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatDevice")){
                    test<swfft<AllToAllGPU,CPUMPI,gpuFFT>,complexFloatDevice>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                #endif
            }

        } else {
            if(world_rank == 0)printf("Invalid mpi_t (%s)\n",argv[2]);
            return;
        }
        #endif
        #endif
    }

    if (!strcmp(argv[1],"AllToAllCPU")){
        #ifndef SWFFT_ALLTOALL
        if(world_rank == 0)printf("Not compiled with AllToAll!\n");
        return;
        #else
        if (!strcmp(argv[2],"CPUMPI")){

            if (!strcmp(argv[3],"fftw")){
                #ifndef SWFFT_FFTW
                if(world_rank == 0)printf("Not compiled with fftw!\n");
                return;
                #else

                if (!strcmp(argv[4],"complexDoubleHost")){
                    test<swfft<AllToAllCPU,CPUMPI,fftw>,complexDoubleHost>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatHost")){
                    test<swfft<AllToAllCPU,CPUMPI,fftw>,complexFloatHost>(false,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexDoubleDevice")){
                    #ifndef SWFFT_GPU
                    if(world_rank == 0)printf("Not compiled with USE_GPU=true!\n");
                    #else
                    test<swfft<AllToAllCPU,CPUMPI,fftw>,complexDoubleDevice>(false,ngx,ngy,ngz,nreps);
                    #endif
                    return;
                }
                if (!strcmp(argv[4],"complexFloatDevice")){
                    #ifndef SWFFT_GPU
                    if(world_rank == 0)printf("Not compiled with USE_GPU=true!\n");
                    #else
                    test<swfft<AllToAllCPU,CPUMPI,fftw>,complexFloatDevice>(false,ngx,ngy,ngz,nreps);
                    #endif
                    return;
                }

                #endif
            }

            if(!strcmp(argv[3],"gpuFFT")){
                #ifndef SWFFT_GPU
                if(world_rank == 0)printf("Not compiled with SWFFT_GPU!\n");
                return;
                #else
                #ifndef SWFFT_CUFFT
                if(world_rank == 0)printf("Not compiled with cuFFT!\n");
                return;
                #else
                if (!strcmp(argv[4],"complexDoubleHost")){
                    test<swfft<AllToAllCPU,CPUMPI,gpuFFT>,complexDoubleHost>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatHost")){
                    test<swfft<AllToAllCPU,CPUMPI,gpuFFT>,complexFloatHost>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexDoubleDevice")){
                    test<swfft<AllToAllCPU,CPUMPI,gpuFFT>,complexDoubleDevice>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                if (!strcmp(argv[4],"complexFloatDevice")){
                    test<swfft<AllToAllCPU,CPUMPI,gpuFFT>,complexFloatDevice>(k_in_blocks,ngx,ngy,ngz,nreps);
                    return;
                }
                #endif
                #endif
            }

        } else {
            if(world_rank == 0)printf("Invalid mpi_t (%s)\n",argv[2]);
            return;
        }
        #endif
    }
}

int main(int argc, char** argv){

    MPI_Init(NULL,NULL);
    #ifdef SWFFT_GPU
    gpuFree(0);
    #endif

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    if (!((argc == 8) || (argc == 10))){
        if(world_rank == 0)printf("USAGE: %s <distribution> <mpi_type> <fft_backend> <T> <k_in_blocks> <nreps> <ngx> [ngy ngz]\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    int ngx = atoi(argv[7]);
    int ngy = ngx;
    int ngz = ngx;
    if (argc == 10){
        ngy = atoi(argv[8]);
        ngz = atoi(argv[9]);
    }

    bool k_in_blocks = false;
    if (!strcmp(argv[5],"true")){
        k_in_blocks = true;
    }

    int nreps = atoi(argv[6]);

    swfft_init_threads();

    run_benckmark(argv,world_rank,ngx,ngy,ngz,k_in_blocks,nreps);
    
    MPI_Finalize();
    return 0;
}