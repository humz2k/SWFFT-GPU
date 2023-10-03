#include "mpiwrangler.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace SWFFT;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

int n_tests = 0;
int n_passed = 0;

#define IS_TRUE(func,T,MPI_T) { n_tests += 1; int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank); if(world_rank == 0){std::cout << TOSTRING(func) << ":\n   " << TOSTRING(T) << "\n   " << TOSTRING(MPI_T) << "\n   ";} if (!(func<T,MPI_T>())){ if(world_rank == 0){std::cout << "failed on line " << __LINE__ << "\n" << std::endl;}}else{if(world_rank == 0){std::cout  << "passed\n" << std::endl; n_passed += 1;}} }

template<class T>
void base_fill(T* data, int val, int n){
    for (int i = 0; i < n; i++){
        data[i].x = val;
        data[i].y = val;
    }
}

void fill(complexDoubleHost* data, int val, int n){
    base_fill(data,val,n);
}

void fill(complexFloatHost* data, int val, int n){
    base_fill(data,val,n);
}

#ifdef SWFFT_GPU
void fill(complexDoubleDevice* data, int val, int n){
    complexDoubleHost* h_data; swfftAlloc(&h_data,sizeof(complexDoubleHost) * n);
    fill(h_data,val,n);
    gpuMemcpy(data,h_data,sizeof(complexDoubleHost) * n, gpuMemcpyHostToDevice);
}

void fill(complexFloatDevice* data, int val, int n){
    complexFloatHost* h_data; swfftAlloc(&h_data,sizeof(complexFloatHost) * n);
    fill(h_data,val,n);
    gpuMemcpy(data,h_data,sizeof(complexFloatHost) * n, gpuMemcpyHostToDevice);
}
#endif

template<class T>
bool base_test(T* data, int world_size){
    for (int i = 0; i < world_size; i++){
        if ((data[i].x != i) || (data[i].y != i))return false;
    }
    return true;
}

bool test(complexDoubleHost* data, int world_size){
    return base_test(data,world_size);
}

bool test(complexFloatHost* data, int world_size){
    return base_test(data,world_size);
}

#ifdef SWFFT_GPU
bool test(complexDoubleDevice* data, int world_size){
    complexDoubleHost* h_data; swfftAlloc(&h_data,sizeof(complexDoubleHost) * world_size);
    gpuMemcpy(h_data,data,sizeof(complexDoubleHost) * world_size, gpuMemcpyDeviceToHost);
    bool out = test(h_data,world_size);
    swfftFree(h_data);
    return out;
}

bool test(complexFloatDevice* data, int world_size){
    complexFloatHost* h_data; swfftAlloc(&h_data,sizeof(complexFloatHost) * world_size);
    gpuMemcpy(h_data,data,sizeof(complexFloatHost) * world_size, gpuMemcpyDeviceToHost);
    bool out = test(h_data,world_size);
    swfftFree(h_data);
    return out;
}
#endif

template<class T, class MPI_T>
bool test_fftwrangler(){

    MPI_T mpi;

    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    int world_size; MPI_Comm_size(MPI_COMM_WORLD,&world_size);

    T* data; swfftAlloc(&data,world_size*sizeof(T));
    T* scratch; swfftAlloc(&scratch,world_size*sizeof(T));

    fill(data,world_rank,world_size);

    mpi.alltoall(data,scratch,1,MPI_COMM_WORLD);

    int tmp = test(scratch,world_size);
    int out;
    MPI_Allreduce(&tmp,&out,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);

    swfftFree(data);
    swfftFree(scratch);
    return out;
}

int main(){
    MPI_Init(NULL,NULL);
    #ifdef SWFFT_GPU
    gpuFree(0);
    #endif

    IS_TRUE(test_fftwrangler,complexDoubleHost,CPUMPI);
    IS_TRUE(test_fftwrangler,complexFloatHost,CPUMPI);
    #ifdef SWFFT_GPU
        IS_TRUE(test_fftwrangler,complexDoubleDevice,CPUMPI);
        IS_TRUE(test_fftwrangler,complexFloatDevice,CPUMPI);

        #ifndef SWFFT_NOCUDAMPI
        IS_TRUE(test_fftwrangler,complexDoubleHost,GPUMPI);
        IS_TRUE(test_fftwrangler,complexFloatHost,GPUMPI);
        IS_TRUE(test_fftwrangler,complexDoubleDevice,GPUMPI);
        IS_TRUE(test_fftwrangler,complexFloatDevice,GPUMPI);
        #endif
    #endif

    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    if(world_rank == 0){
        printf("%d/%d tests passed\n",n_passed,n_tests);
    }
    MPI_Finalize();
    return 0;
}


