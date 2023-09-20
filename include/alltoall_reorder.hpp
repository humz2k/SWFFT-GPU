#ifdef ALLTOALL
#ifndef ALLTOALL_REORDER_SEEN
#define ALLTOALL_REORDER_SEEN

#include <stdlib.h>
#include <stdio.h>

#include "complex-type.h"
#include "gpu.hpp"

namespace A2A{
    #ifdef GPU
    class GPUReorder{
        public:
            int3 ng;
            int3 dims;
            int3 coords;
            int3 local_grid;
            int local_grid_size[3];
            int nlocal;
            int world_size;
            int blockSize;

            GPUReorder();
            GPUReorder(int3 ng_, int3 dims_, int3 coords_, int blockSize_);
            ~GPUReorder();

            template<class T>
            void gpu_shuffle_indices(T* Buff1, T* Buff2, int n);

            void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n);
            void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n);
            void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n);
            void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n);

            template<class T>
            void gpu_reorder(T* Buff1, T* Buff2, int n, int direction);

            void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n, int direction);
            void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n, int direction);
            void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n, int direction);
            void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n, int direction);
    };
    #endif

    class CPUReorder{
        public:
            int3 ng;
            int3 dims;
            int3 coords;
            int3 local_grid;
            int local_grid_size[3];
            int nlocal;
            int world_size;
            int blockSize;

            CPUReorder();
            CPUReorder(int3 ng_, int3 dims_, int3 coords_, int blockSize_);
            ~CPUReorder();

            template<class T>
            void cpu_shuffle_indices(T* Buff1, T* Buff2, int n);

            #ifdef GPU
            void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n);
            void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n);
            #endif
            void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n);
            void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n);

            template<class T>
            void cpu_reorder(T* Buff1, T* Buff2, int n, int direction);

            #ifdef GPU
            void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n, int direction);
            void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n, int direction);
            #endif
            void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n, int direction);
            void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n, int direction);
    };
}

#endif
#endif