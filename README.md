# SWFFT-GPU (WIP)

## Minimal example

```
#include <stdio.h>
#include <stdlib.h>
#include "swfft.hpp"

int main(){
    swfft<AllToAllGPU,CPUMPI,gpuFFT> my_swfft(MPI_COMM_WORLD,256,256,256,64);

    complexDoubleDevice* data; swfftAlloc(&data,sizeof(complexDoubleDevice) * my_swfft.buff_sz());
    complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * my_swfft.buff_sz());

    my_swfft.forward(data,scratch);
    my_swfft.backward(data,scratch);

    swfftFree(data);
    swfftFree(scratch);

    return 0;
}
```