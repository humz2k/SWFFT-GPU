#ifdef ALLTOALL
#ifndef ALLTOALL_SEEN
#define ALLTOALL_SEEN

#include "alltoall_reorder.hpp"
#include <mpi.h>
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"

namespace A2A{

    template<class MPI_T, class REORDER_T>
    class Distribution{
        public:
            int ndims;
            int ng[3];
            int nlocal;
            int world_size;
            int world_rank;
            int local_grid_size[3];
            int dims[3];
            int coords[3];
            int local_coordinates_start[3];
            MPI_Comm comm;
            MPI_Comm fftcomms[3];

            MPI_T mpi;
            REORDER_T reordering;

            int blockSize;

            Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
            ~Distribution();

            MPI_Comm shuffle_comm_1();
            MPI_Comm shuffle_comm_2();
            MPI_Comm shuffle_comm(int n);

            template<class T>
            inline void getPencils_(T* Buff1, T* Buff2, int dim);
            
            void getPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int dim);
            void getPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int dim);
            void getPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int dim);
            void getPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

            template<class T>
            inline void returnPencils_(T* Buff1, T* Buff2, int dim);

            void returnPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int dim);
            void returnPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int dim);
            void returnPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int dim);
            void returnPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

            void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n);
            void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n);
            void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n);
            void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n);

            void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n, int direction);
            void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n, int direction);
            void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n, int direction);
            void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n, int direction);

    };

}

#endif
#endif