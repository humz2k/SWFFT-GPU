#ifdef SWFFT_ALLTOALL
#include "alltoall.hpp"

namespace SWFFT{
namespace A2A{

    /*
    Calculates the coordinates of the rank from the dimensions from MPI_Dims_create.
    */
    void rank2coords(int rank, int* dims, int* out){

    out[0] = rank / (dims[1]*dims[2]);
    out[1] = (rank - out[0]*(dims[1]*dims[2])) / dims[2];
    out[2] = rank - out[0]*(dims[1]*dims[2]) - out[1]*(dims[2]);

    }

    /*
    Gets the dimensions of the local grid of particles from the dimensions from MPI_Dims_create.
    */
    void topology2localgrid(int Ng, int* dims, int* grid_size){

        grid_size[0] = Ng / dims[0];
        grid_size[1] = Ng / dims[1];
        grid_size[2] = Ng / dims[2];

    }

    void topology2localgrid(int ngx, int ngy, int ngz, int* dims, int* grid_size){

        grid_size[0] = ngx / dims[0];
        grid_size[1] = ngy / dims[1];
        grid_size[2] = ngz / dims[2];

    }

    /*
    Gets the global coordinates of the starting point of the local grid.
    */
    void get_local_grid_start(int* local_grid_size, int* coords, int* local_coordinates_start){

        local_coordinates_start[0] = local_grid_size[0] * coords[0];
        local_coordinates_start[1] = local_grid_size[1] * coords[1];
        local_coordinates_start[2] = local_grid_size[2] * coords[2];

    }

    #ifdef SWFFT_GPU
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::copy(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2){
        gpuMemcpy(Buff1,Buff2,sizeof(complexDoubleDevice) * nlocal,gpuMemcpyDeviceToDevice);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::copy(complexFloatDevice* Buff1, complexFloatDevice* Buff2){
        gpuMemcpy(Buff1,Buff2,sizeof(complexFloatDevice) * nlocal,gpuMemcpyDeviceToDevice);
    }
    #endif
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::copy(complexDoubleHost* __restrict Buff1, const complexDoubleHost* __restrict Buff2){
        for (int i = 0; i < nlocal; i++){
            Buff1[i] = Buff2[i];
        }
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::copy(complexFloatHost* __restrict Buff1, const complexFloatHost* __restrict Buff2){
        for (int i = 0; i < nlocal; i++){
            Buff1[i] = Buff2[i];
        }
    }

    template<class MPI_T, class REORDER_T>
    MPI_Comm Distribution<MPI_T,REORDER_T>::shuffle_comm_1(){
        int new_rank = coords[2] * dims[0] * dims[1] + coords[0] * dims[1] + coords[1];
        MPI_Comm new_comm;

        MPI_Comm_split(comm,0,new_rank,&new_comm);

        return new_comm;
    }

    template<class MPI_T, class REORDER_T>
    MPI_Comm Distribution<MPI_T,REORDER_T>::shuffle_comm_2(){
        int new_rank = coords[1] * dims[2] * dims[0] + coords[2] * dims[0] + coords[0];
        MPI_Comm new_comm;

        MPI_Comm_split(comm,0,new_rank,&new_comm);

        return new_comm;
    }

    template<class MPI_T, class REORDER_T>
    MPI_Comm Distribution<MPI_T,REORDER_T>::shuffle_comm(int start){
        if (start == 0)return shuffle_comm_2();
        if (start == 1)return shuffle_comm_1();
        if (start == 2)return comm;
        return comm;
    }

    template<class MPI_T, class REORDER_T>
    Distribution<MPI_T,REORDER_T>::Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_, bool ks_as_block_) : comm(comm_), blockSize(blockSize_), dims{0,0,0}, ks_as_block(ks_as_block_){
        ng[0] = ngx;
        ng[1] = ngy;
        ng[2] = ngz;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);
        ndims = 3;
        dims[0] = 0;
        dims[1] = 0;
        dims[2] = 0;
        MPI_Dims_create(world_size,ndims,dims);

        rank2coords(world_rank,dims,coords);

        topology2localgrid(ng[0],ng[1],ng[2],dims,local_grid_size);
        get_local_grid_start(local_grid_size,coords,local_coordinates_start);

        for (int i = 0; i < 3; i++){
            fftcomms[i] = shuffle_comm(i);
        }

        nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

        REORDER_T reordering_(make_int3(ng[0],ng[1],ng[2]),make_int3(dims[0],dims[1],dims[2]),make_int3(coords[0],coords[1],coords[2]),blockSize);

        reordering = reordering_;
    }

    template<class MPI_T, class REORDER_T>
    Distribution<MPI_T,REORDER_T>::~Distribution(){
        for (int i = 0; i < 2; i++){
            MPI_Comm_free(&fftcomms[i]);
        }
    };

    template<class MPI_T, class REORDER_T>
    template<class T>
    inline void Distribution<MPI_T,REORDER_T>::getPencils_(T* Buff1, T* Buff2, int n){
        int dim = (n+2)%3;
        
        MPI_Comm my_comm = fftcomms[dim];

        int nsends = (nlocal / world_size);

        mpi.alltoall(Buff1,Buff2,nsends,my_comm);
    }

    #ifdef SWFFT_GPU
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::getPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n){
        getPencils_(Buff1,Buff2,n);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::getPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n){
        getPencils_(Buff1,Buff2,n);
    }
    #endif

    template<class MPI_T, class REORDER_T>   
    void Distribution<MPI_T,REORDER_T>::getPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n){
        getPencils_(Buff1,Buff2,n);
    }
    
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::getPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int n){
        getPencils_(Buff1,Buff2,n);
    }

    template<class MPI_T, class REORDER_T>
    template<class T>
    inline void Distribution<MPI_T,REORDER_T>::returnPencils_(T* Buff1, T* Buff2, int n){
        int dim = (n+2)%3;
        
        MPI_Comm my_comm = fftcomms[dim];

        int nsends = (nlocal / world_size);

        mpi.alltoall(Buff1,Buff2,nsends,my_comm);
    }

    #ifdef SWFFT_GPU
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::returnPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n){
        returnPencils_(Buff1,Buff2,n);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::returnPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n){
        returnPencils_(Buff1,Buff2,n);
    }
    #endif

    template<class MPI_T, class REORDER_T>   
    void Distribution<MPI_T,REORDER_T>::returnPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n){
        returnPencils_(Buff1,Buff2,n);
    }
    
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::returnPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int n){
        returnPencils_(Buff1,Buff2,n);
    }

    #ifdef SWFFT_GPU
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n){
        reordering.shuffle_indices(Buff1,Buff2,n);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n){
        reordering.shuffle_indices(Buff1,Buff2,n);
    }
    #endif

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n){
        reordering.shuffle_indices(Buff1,Buff2,n);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n){
        reordering.shuffle_indices(Buff1,Buff2,n);
    }

    #ifdef SWFFT_GPU
    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n, int direction){
        reordering.reorder(Buff1,Buff2,n, direction);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n, int direction){
        reordering.reorder(Buff1,Buff2,n, direction);
    }
    #endif

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n, int direction){
        reordering.reorder(Buff1,Buff2,n, direction);
    }

    template<class MPI_T, class REORDER_T>
    void Distribution<MPI_T,REORDER_T>::reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n, int direction){
        reordering.reorder(Buff1,Buff2,n, direction);
    }
}
}
    template class SWFFT::A2A::Distribution<SWFFT::CPUMPI,SWFFT::A2A::CPUReorder>;
    #ifdef SWFFT_GPU
    template class SWFFT::A2A::Distribution<SWFFT::CPUMPI,SWFFT::A2A::GPUReorder>;
    #ifndef SWFFT_NOCUDAMPI
    template class SWFFT::A2A::Distribution<SWFFT::GPUMPI,SWFFT::A2A::CPUReorder>;
    template class SWFFT::A2A::Distribution<SWFFT::GPUMPI,SWFFT::A2A::GPUReorder>;
    #endif
    #endif


#endif