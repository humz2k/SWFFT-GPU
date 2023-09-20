#ifndef MPI_WRANGLER_INCLUDED
#define MPI_WRANGLER_INCLUDED
#include <mpi.h>

#include "gpu.hpp"

#include "complex-type.h"

class CPUMPI{
    public:
        CPUMPI();
        ~CPUMPI();

        #ifdef GPU
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