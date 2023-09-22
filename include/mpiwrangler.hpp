#ifndef SWFFT_MPI_WRANGLER_INCLUDED
#define SWFFT_MPI_WRANGLER_INCLUDED
#include <mpi.h>

#include "gpu.hpp"

#include "complex-type.h"

namespace SWFFT{

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

        #ifdef SWFFT_GPU
        template<class T>
        void gpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm);
        
        void alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm);
        #endif

        void alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm);

        /*#ifdef GPU
        template<class T>
        void gpu_memcpy_irecv(T* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req);

        template<class T>
        void gpu_memcpy_isend(T* buff, int count, int dest, int tag, MPI_Comm comm, MPI_Request* req);

        void irecv(complexDoubleDevice* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req);
        void irecv(complexFloatDevice* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req);

        void isend(complexDoubleDevice* buff, int count, int dest, int tag, MPI_Comm comm, MPI_Request* req);
        void isend(complexFloatDevice* buff, int count, int dest, int tag, MPI_Comm comm, MPI_Request* req);
        #endif

        void irecv(complexDoubleHost* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req);
        void irecv(complexFloatHost* buff, int count, int source, int tag, MPI_Comm comm, MPI_Request* req);

        void isend(complexDoubleHost* buff, int count, int dest, int tag, MPI_Comm comm, MPI_Request* req);
        void isend(complexFloatHost* buff, int count, int dest, int tag, MPI_Comm comm, MPI_Request* req);*/

};

#ifdef SWFFT_GPU
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
}
#endif