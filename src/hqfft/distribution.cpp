#ifdef SWFFT_HQFFT

#include "hqfft.hpp"

namespace SWFFT{
namespace HQFFT{

template<template<class> class Communicator, class MPI_T, class REORDER_T>
Distribution<Communicator,MPI_T,REORDER_T>::Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_) : world_comm(comm_),ng{ngx,ngy,ngz},blockSize(blockSize_){
    MPI_Comm_rank(world_comm,&world_rank);
    MPI_Comm_size(world_comm,&world_size);
    
    MPI_Dims_create(world_size,3,dims);

    local_grid_size[0] = ng[0] / dims[0];
    local_grid_size[1] = ng[1] / dims[1];
    local_grid_size[2] = ng[2] / dims[2];

    nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    coords[0] = world_rank / (dims[1] * dims[2]);
    coords[1] = (world_rank - coords[0] * (dims[1] * dims[2])) / dims[2];
    coords[2] = (world_rank - coords[0] * (dims[1] * dims[2])) - coords[1] * dims[2];

    local_coords_start[0] = local_grid_size[0] * coords[0];
    local_coords_start[1] = local_grid_size[1] * coords[1];
    local_coords_start[2] = local_grid_size[2] * coords[2];

    if (world_rank == 0){
        printf("Distribution:\n");
        printf("   ng              = [%d %d %d]\n",ng[0],ng[1],ng[2]);
        printf("   dims            = [%d %d %d]\n",dims[0],dims[1],dims[2]);
        printf("   local_grid_size = [%d %d %d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("   nlocal          = %d\n",nlocal);
        printf("   blockSize       = %d\n",blockSize);
        printf("   world_size      = %d\n",world_size);
        #ifndef nocudampi
        printf("   using cuda mpi\n");
        #else
        printf("   not using cuda mpi (!!)\n");
        #endif
        
    }
    
    int z_col_idx = coords[0] * dims[1] + coords[1];
    int z_col_rank = coords[2];

    MPI_Comm_split(world_comm,z_col_idx,z_col_rank,&distcomms[0]);

    int y_col_idx = coords[0] * dims[2] + coords[2];
    int y_col_rank = coords[1];

    MPI_Comm_split(world_comm,y_col_idx,y_col_rank,&distcomms[1]);

    int x_col_idx = coords[1];
    int x_col_rank = coords[0] * dims[2] + coords[2];

    MPI_Comm_split(world_comm,x_col_idx,x_col_rank,&distcomms[2]);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
Distribution<Communicator,MPI_T,REORDER_T>::~Distribution(){
    MPI_Comm_free(&distcomms[0]);
    MPI_Comm_free(&distcomms[1]);
    MPI_Comm_free(&distcomms[2]);
}


template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::reshape_1(T* buff1, T* buff2){
    int n_recvs = dims[2];
    int mini_pencil_size = local_grid_size[2];
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    reorder.reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::unreshape_1(T* buff1, T* buff2){
    int z_dim = ng[2];
    int x_dim = local_grid_size[0] / dims[2];
    int y_dim = (nlocal / z_dim) / x_dim;

    reorder.unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::reshape_2(T* buff1, T* buff2){
    int n_recvs = dims[1];
    int mini_pencil_size = local_grid_size[1];
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    reorder.reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::unreshape_2(T* buff1, T* buff2){
    int z_dim = ng[1];
    int x_dim = local_grid_size[2] / dims[1];
    int y_dim = (nlocal / z_dim) / x_dim;

    reorder.unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::reshape_3(T* buff1, T* buff2){
    int n_recvs = dims[0] * dims[2];
    int mini_pencil_size = ng[0] / n_recvs;
    int send_per_rank = nlocal / n_recvs;
    int pencils_per_rank = send_per_rank / mini_pencil_size;
    reorder.reshape(buff1,buff2,n_recvs,mini_pencil_size,send_per_rank,pencils_per_rank,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::unreshape_3(T* buff1, T* buff2){
    int z_dim = ng[0];
    int x_dim = local_grid_size[1] / dims[0];
    int y_dim = (nlocal / z_dim) / x_dim;

    reorder.unreshape(buff1,buff2,z_dim,x_dim,y_dim,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::reshape_final(T* buff1, T* buff2, int ny, int nz){
    reorder.reshape_final(buff1,buff2,ny,nz,local_grid_size,nlocal,blockSize);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::_pencils_1(T* buff1, T* buff2){
    CollectiveComm.alltoall(buff1,buff2,(nlocal / dims[2]),distcomms[0]);
    reshape_1(buff1,buff2);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::_pencils_2(T* buff1, T* buff2){
    unreshape_1(buff1,buff2);
    CollectiveComm.alltoall(buff2,buff1,(nlocal / dims[1]),distcomms[1]);
    reshape_2(buff1,buff2);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::_pencils_3(T* buff1, T* buff2){
    unreshape_2(buff1,buff2);
    CollectiveComm.alltoall(buff2,buff1,(nlocal / (dims[2] * dims[0])),distcomms[2]);
    reshape_3(buff1,buff2);
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
template<class T>
void Distribution<Communicator,MPI_T,REORDER_T>::_return_pencils(T* buff1, T* buff2){
    unreshape_3(buff1,buff2);

    int dest_x_start = 0;
    int dest_x_end = dims[0] - 1;

    int y = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) / local_grid_size[1];

    int y_send = local_grid_size[1] / (ng[1] / (dims[0] * dims[2]));
    int y_send_id = (((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2]))) % local_grid_size[1]) / y_send;

    int z = (coords[1] * (ng[2] / dims[1])) / local_grid_size[2];

    int n_recvs = dims[0];

    CPUIsend<T> isends[n_recvs];
    CPUIrecv<T> irecvs[n_recvs];

    int count = 0;
    for (int x = dest_x_start; x < dest_x_end+1; x++){
        int dest = x*dims[1]*dims[2] + y*dims[2] + z;

        int reqidx = x - dest_x_start;
        int nx = (dest_x_end+1) - dest_x_start;
        int ny = n_recvs / nx;

        int xsrc = x * local_grid_size[0];
        int ysrc = ((coords[0] * dims[2] + coords[2]) * (ng[1] / (dims[0] * dims[2])));
        int zsrc = (coords[1] * (ng[2] / dims[1]));
        int id = xsrc * local_grid_size[1] * local_grid_size[2] + ysrc * local_grid_size[2] + zsrc;

        int tmp1 = count / y_send;

        int xoff = 0;
        int yoff = (count - tmp1 * y_send) * (ng[1] / (dims[0] * dims[2]));
        int zoff = tmp1 * (ng[2] / dims[1]);

        int xrec = local_grid_size[0] * coords[0] + xoff;
        int yrec = local_grid_size[1] * coords[1] + yoff;
        int zrec = local_grid_size[2] * coords[2] + zoff;
        int recid = xrec * local_grid_size[1] * local_grid_size[2] + yrec * local_grid_size[2] + zrec;

        isends[count] = CollectiveComm.mpi.isend(&buff2[count*(nlocal/n_recvs)],(nlocal/n_recvs),dest,id,world_comm);
        irecvs[count] = CollectiveComm.mpi.irecv(&buff1[count*(nlocal/n_recvs)],(nlocal/n_recvs),MPI_ANY_SOURCE,recid,world_comm);

        count++;

    }

    for (int i = 0; i < n_recvs; i++){
        isends[i].execute();
        irecvs[i].execute();
    }

    for (int i = 0; i < n_recvs; i++){
        isends[i].wait();
        irecvs[i].wait();
    }

    for (int i = 0; i < n_recvs; i++){
        irecvs[i].finalize();
    }

    reshape_final(buff1,buff2,y_send,n_recvs / y_send);
}
#ifdef SWFFT_GPU
template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_1(complexDoubleDevice* buff1, complexDoubleDevice* buff2){
    _pencils_1(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_1(complexFloatDevice* buff1, complexFloatDevice* buff2){
    _pencils_1(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_2(complexDoubleDevice* buff1, complexDoubleDevice* buff2){
    _pencils_2(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_2(complexFloatDevice* buff1, complexFloatDevice* buff2){
    _pencils_2(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_3(complexDoubleDevice* buff1, complexDoubleDevice* buff2){
    _pencils_3(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_3(complexFloatDevice* buff1, complexFloatDevice* buff2){
    _pencils_3(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::return_pencils(complexDoubleDevice* buff1, complexDoubleDevice* buff2){
    return_pencils(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::return_pencils(complexFloatDevice* buff1, complexFloatDevice* buff2){
    return_pencils(buff1,buff2);   
}
#endif

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_1(complexDoubleHost* buff1, complexDoubleHost* buff2){
    _pencils_1(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_1(complexFloatHost* buff1, complexFloatHost* buff2){
    _pencils_1(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_2(complexDoubleHost* buff1, complexDoubleHost* buff2){
    _pencils_2(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_2(complexFloatHost* buff1, complexFloatHost* buff2){
    _pencils_2(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_3(complexDoubleHost* buff1, complexDoubleHost* buff2){
    _pencils_3(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::pencils_3(complexFloatHost* buff1, complexFloatHost* buff2){
    _pencils_3(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::return_pencils(complexDoubleHost* buff1, complexDoubleHost* buff2){
    return_pencils(buff1,buff2);   
}

template<template<class> class Communicator, class MPI_T, class REORDER_T>
void Distribution<Communicator,MPI_T,REORDER_T>::return_pencils(complexFloatHost* buff1, complexFloatHost* buff2){
    return_pencils(buff1,buff2);   
}

}
}

#endif