#include "swfftc.h"

int main(){

    MPI_Init(NULL,NULL);

    #ifdef SWFFT_GPU
    #ifdef SWFFT_CUFFT
    #ifdef SWFFT_ALLTOALL

    SWFFT__AllToAllGPU_CPUMPI_gpuFFT* my_swfft = SWFFT__AllToAllGPU_CPUMPI_gpuFFT__new(MPI_COMM_WORLD,8,8,8,64,1);

    SWFFT__AllToAllGPU_CPUMPI_gpuFFT__query(my_swfft);

    int buff_sz = SWFFT__AllToAllGPU_CPUMPI_gpuFFT__buff_sz(my_swfft);

    complexDoubleDevice* data; swfftAlloc_complexDoubleDevice(&data,sizeof(complexDoubleDevice) * buff_sz);
    complexDoubleDevice* scratch; swfftAlloc_complexDoubleDevice(&scratch,sizeof(complexDoubleDevice) * buff_sz);

    swfftFree_complexDoubleDevice(data);
    swfftFree_complexDoubleDevice(scratch);

    SWFFT__AllToAllGPU_CPUMPI_gpuFFT__del(my_swfft);

    #endif
    #endif
    #endif

    MPI_Finalize();

    return 0;
}