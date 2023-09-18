#ifdef ALLTOALL
#include "alltoall.hpp"

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
    Distribution<MPI_T,REORDER_T>::Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_) : comm(comm_), blockSize(blockSize_){
        ng[0] = ngx;
        ng[1] = ngy;
        ng[2] = ngz;
        MPI_Comm_size(comm, &world_size);
        MPI_Comm_rank(comm, &world_rank);

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

        reordering(make_int3(ng[0],ng[1],ng[2]),make_int3(dims[0],dims[1],dims[2]),make_int3(coords[0],coords[1],coords[2]),blockSize);

    }

    template<class MPI_T, class REORDER_T>
    Distribution<MPI_T,REORDER_T>::~Distribution(){};

    template<class MPI_T, class REORDER_T>
    template<class T>
    void Distribution<MPI_T,REORDER_T>::getPencils(T* Buff1, T* Buff2, int dim){
        
    }
}

#endif