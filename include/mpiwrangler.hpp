#ifndef SWFFT_MPI_WRANGLER_INCLUDED
#define SWFFT_MPI_WRANGLER_INCLUDED
#include <mpi.h>

#include "gpu.hpp"

#include "complex-type.h"

namespace SWFFT{

template<class T>
class CPUIsend{
    private:
        void* h_in_buff;
        MPI_Request req;
        bool initialized;
        T* in_buff;
        int n;
        int dest;
        int tag;
        MPI_Comm comm;
        #ifdef SWFFT_GPU
        gpuEvent_t event;
        #endif
    
    public:
        CPUIsend();
        CPUIsend(T* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_);
        ~CPUIsend();

        void execute();

        void wait();

};

template<class T>
class CPUIrecv{
    private:
        void* h_out_buff;
        T* out_buff;
        MPI_Request req;
        size_t sz;
        bool initialized;
        int n;
        int source;
        int tag;
        MPI_Comm comm;
        #ifdef SWFFT_GPU
        gpuEvent_t event;
        #endif

    public:
        CPUIrecv();
        CPUIrecv(T* my_out_buff, int n, int source, int tag, MPI_Comm comm);
        ~CPUIrecv();

        void execute();

        void wait();

        void finalize();

};

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

        void query();

        #ifdef SWFFT_GPU
        template<class T>
        void gpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm);
        
        void alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm);
        #endif

        void alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm);

        #ifdef SWFFT_GPU
        CPUIsend<complexDoubleDevice> isend(complexDoubleDevice* buff, int n, int dest, int tag, MPI_Comm comm);
        CPUIsend<complexFloatDevice> isend(complexFloatDevice* buff, int n, int dest, int tag, MPI_Comm comm);
        #endif

        CPUIsend<complexDoubleHost> isend(complexDoubleHost* buff, int n, int dest, int tag, MPI_Comm comm);
        CPUIsend<complexFloatHost> isend(complexFloatHost* buff, int n, int dest, int tag, MPI_Comm comm);

        #ifdef SWFFT_GPU
        CPUIrecv<complexDoubleDevice> irecv(complexDoubleDevice* buff, int n, int dest, int tag, MPI_Comm comm);
        CPUIrecv<complexFloatDevice> irecv(complexFloatDevice* buff, int n, int dest, int tag, MPI_Comm comm);
        #endif

        CPUIrecv<complexDoubleHost> irecv(complexDoubleHost* buff, int n, int dest, int tag, MPI_Comm comm);
        CPUIrecv<complexFloatHost> irecv(complexFloatHost* buff, int n, int dest, int tag, MPI_Comm comm);

        #ifdef SWFFT_GPU
        void sendrecv(complexDoubleDevice* send_buff, int sendcount, int dest, int sendtag, complexDoubleDevice* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm);
        void sendrecv(complexFloatDevice* send_buff, int sendcount, int dest, int sendtag, complexFloatDevice* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm);
        #endif
        
        void sendrecv(complexDoubleHost* send_buff, int sendcount, int dest, int sendtag, complexDoubleHost* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm);
        void sendrecv(complexFloatHost* send_buff, int sendcount, int dest, int sendtag, complexFloatHost* recv_buff, int recvcount, int source, int recvtag, MPI_Comm comm);

};

#ifdef SWFFT_GPU
#ifndef nocudampi
class GPUMPI{
    public:
        GPUMPI();
        ~GPUMPI();

        void query();

        void alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n, MPI_Comm comm);

        void alltoall(complexDoubleHost* buff1, complexDoubleHost* buff2, int n, MPI_Comm comm);
        void alltoall(complexFloatHost* buff1, complexFloatHost* buff2, int n, MPI_Comm comm);
};
#endif
#endif
}
#endif