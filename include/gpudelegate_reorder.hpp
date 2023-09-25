#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
#ifdef SWFFT_GPUDELEGATE
#ifndef SWFFT_GPUDELEGATE_REORDER_SEEN
#define SWFFT_GPUDELEGATE_REORDER_SEEN

#include "gpu.hpp"
#include "complex-type.h"

namespace SWFFT{

namespace GPUDELEGATE{


    void reorder(int direction, complexDoubleDevice* buff1, complexDoubleDevice* buff2, int3 ng, int3 local_grid_size, int3 dims, int rank, gpuStream_t stream, int blockSize);
    void reorder(int direction, complexFloatDevice* buff1, complexFloatDevice* buff2, int3 ng, int3 local_grid_size, int3 dims, int rank, gpuStream_t stream, int blockSize);

}

}


#endif
#endif
#endif
#endif