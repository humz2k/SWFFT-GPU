#ifndef MPI_WRANGLER_INCLUDED
#define MPI_WRANGLER_INCLUDED
#include <mpi.h>

#include "gpu.hpp"

#include "complex-type.h"

class CPUMPI{
    private:
        void* _h_buff1;
        void* _h_buff2;
        size_t last_size;

        void* get_h_buff1(size_t sz);
        void* get_h_buff2(size_t sz);

    public:
        CPUMPI();
        ~CPUMPI();

        #ifdef GPU
        template<class T>
        void gpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm);
        
        void alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm);
        #endif

        void alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm);

};

#ifdef GPU
#ifndef nocudampi
class GPUMPI{
    public:
        GPUMPI();
        ~GPUMPI();

        void alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm);

        void alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm);
};
#endif
#endif

#endif