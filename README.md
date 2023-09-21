# SWFFT-GPU (WIP)

## Minimal example

```
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "swfft.hpp"

int test(){
    swfft<AllToAllGPU,CPUMPI,gpuFFT> my_swfft(MPI_COMM_WORLD,256,256,256,64);

    complexDoubleDevice* data; swfftAlloc(&data,sizeof(complexDoubleDevice) * my_swfft.buff_sz());
    complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * my_swfft.buff_sz());

    int3 this_position_in_kspace = my_swfft.get_ks(0);
    printf("index 0 is at (%d %d %d) in kspace\n",this_position_in_kspace.x,this_position_in_kspace.y,this_position_in_kspace.z);

    my_swfft.forward(data,scratch);
    my_swfft.backward(data,scratch);

    swfftFree(data);
    swfftFree(scratch);
}

int main(){

    MPI_Init(NULL,NULL);

    test();

    MPI_Finalize();

    return 0;
}
```