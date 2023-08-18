#include "alltoall.hpp"

#ifdef ALLTOALL

#include "a2atopencils.hpp"
#include "changefastaxis.hpp"

#define CheckCondition(cond) if((!(cond)) && (world_rank == 0)){printf("Distribution: Failed Check (%s)\n",TOSTRING(cond)); MPI_Abort(MPI_COMM_WORLD,1);}

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

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

/*
Gets the global coordinates of the starting point of the local grid.
*/
void get_local_grid_start(int* local_grid_size, int* coords, int* local_coordinates_start){

    local_coordinates_start[0] = local_grid_size[0] * coords[0];
    local_coordinates_start[1] = local_grid_size[1] * coords[1];
    local_coordinates_start[2] = local_grid_size[2] * coords[2];

}

template<class T>
MPI_Comm Distribution<T>::shuffle_comm_1(){
    int new_rank = coords[2] * dims[0] * dims[1] + coords[0] * dims[1] + coords[1];
    MPI_Comm new_comm;

    MPI_Comm_split(comm,0,new_rank,&new_comm);

    return new_comm;
}

template<class T>
MPI_Comm Distribution<T>::shuffle_comm_2(){
    int new_rank = coords[1] * dims[2] * dims[0] + coords[2] * dims[0] + coords[0];
    MPI_Comm new_comm;

    MPI_Comm_split(comm,0,new_rank,&new_comm);

    return new_comm;
}

template<class T>
MPI_Comm Distribution<T>::shuffle_comm(int start){
    if (start == 0)return shuffle_comm_2();
    if (start == 1)return shuffle_comm_1();
    if (start == 2)return comm;
    return comm;
}



template<class T>
Distribution<T>::Distribution(MPI_Comm input_comm, int input_Ng, int input_blockSize, int nBatches): 
            dims {0,0,0}, coords {0,0,0}, local_grid_size {0,0,0}, local_coordinates_start {0,0,0}, batches(nBatches){

    gpuStreamCreate(&diststream);

    ndims = 3; //Number of dimensions
    Ng = input_Ng; //Ng
    comm = input_comm; //communicator

    //get the world_size, world_rank and dims
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);
    MPI_Dims_create(world_size,ndims,dims);


    rank2coords(world_rank,dims,coords);
    topology2localgrid(Ng,dims,local_grid_size);
    get_local_grid_start(local_grid_size,coords,local_coordinates_start);
    
    for (int i = 0; i < 3; i++){
        fftcomms[i] = shuffle_comm(i);
    }


    CheckCondition((Ng % dims[0]) == 0);
    CheckCondition((Ng % dims[1]) == 0);
    CheckCondition((Ng % dims[2]) == 0);

    CheckCondition((world_size % dims[0]) == 0);
    CheckCondition((world_size % dims[1]) == 0);
    CheckCondition((world_size % dims[2]) == 0);

    CheckCondition(((local_grid_size[0] * local_grid_size[1]) % world_size) == 0);
    CheckCondition(((local_grid_size[0] * local_grid_size[2]) % world_size) == 0);
    CheckCondition(((local_grid_size[1] * local_grid_size[2]) % world_size) == 0);


    //calculate number of local grid cells
    nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    blockSize = input_blockSize;
    numBlocks = (nlocal + blockSize - 1) / blockSize;

    //allocate some scratch space to do swaps to host from device if not using cuda-aware MPI
    #ifdef nocudampi
    h_scratch1 = (T*)malloc(nlocal*sizeof(T));
    h_scratch2 = (T*)malloc(nlocal*sizeof(T));
    #endif

    MPI_Type_contiguous((int)sizeof(T), MPI_BYTE, &TYPE_COMPLEX); 
    MPI_Type_commit(&TYPE_COMPLEX);

    if (world_rank == 0){
        printf("USING ALLTOALL BACKEND\n");
        #ifdef cudampi
            printf("USING CUDA-AWARE MPI\n");
        #else
            printf("NOT USING CUDA-AWARE MPI\n");
        #endif
        printf("#######\nDISTRIBUTION PARAMETERS\n");
        printf("DIMS [%d,%d,%d]\n",dims[0],dims[1],dims[2]);
        printf("LOCAL_GRID_SIZE [%d,%d,%d]\n",local_grid_size[0],local_grid_size[1],local_grid_size[2]);
        printf("NLOCAL: %d\n",nlocal);
        printf("#######\n\n");
    }


}

template<class T>
void Distribution<T>::shuffle_indices(T* Buff1, T* Buff2, int n){
    switch(n){
        case 0:
            launch_fast_z_to_x(Buff2, Buff1, local_grid_size, blockSize, numBlocks, nlocal);
            break;
        case 1:
            launch_fast_x_to_y(Buff2, Buff1, local_grid_size, blockSize, numBlocks, nlocal);
            break;
        case 2:
            launch_fast_y_to_z(Buff2, Buff1, local_grid_size, blockSize, numBlocks, nlocal);
            break;
    }
}

template<class T>
void Distribution<T>::s_alltoall_forward(T* h_src, T* h_dest, int n, int pencil_size, MPI_Comm comm_){
    int comm_size;
    int comm_rank;
    MPI_Comm_size(comm_,&comm_size);
    MPI_Comm_rank(comm_,&comm_rank);

    int pencils_per_send = n / pencil_size;
    int stride = comm_size * pencil_size;

    MPI_Datatype strided_t;
    MPI_Type_vector(pencils_per_send,pencil_size * sizeof(T),stride * sizeof(T),MPI_BYTE,&strided_t);

    MPI_Type_commit(&strided_t);

    MPI_Request send_requests[comm_size];
    MPI_Request recv_requests[comm_size];

    for (int i = 0; i < comm_size; i++){
        MPI_Irecv(&h_dest[i*pencil_size],1,strided_t,i,0,comm_,&recv_requests[i]);
    }
    for (int i = 0; i < comm_size; i++){
        MPI_Isend(&h_src[i*n],n * sizeof(T),MPI_BYTE,i,0,comm_,&send_requests[i]);
    }

    for (int i = 0; i < comm_size; i++){
        MPI_Wait(&send_requests[i],MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < comm_size; i++){
        MPI_Wait(&recv_requests[i],MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&strided_t);

}

template<class T>
void Distribution<T>::s_alltoall_backward(T* h_src, T* h_dest, int n, int pencil_size, MPI_Comm comm_){
    int comm_size;
    int comm_rank;
    MPI_Comm_size(comm_,&comm_size);
    MPI_Comm_rank(comm_,&comm_rank);

    int pencils_per_send = n / pencil_size;
    int stride = comm_size * pencil_size;

    MPI_Datatype strided_t;
    MPI_Type_vector(pencils_per_send,pencil_size * sizeof(T),stride * sizeof(T),MPI_BYTE,&strided_t);

    MPI_Type_commit(&strided_t);

    MPI_Request send_requests[comm_size];
    MPI_Request recv_requests[comm_size];

    for (int i = 0; i < comm_size; i++){
        MPI_Irecv(&h_dest[i*n],n * sizeof(T),MPI_BYTE,i,0,comm_,&recv_requests[i]);
    }
    for (int i = 0; i < comm_size; i++){
        MPI_Isend(&h_src[i*pencil_size],1,strided_t,i,0,comm_,&send_requests[i]);
    }

    for (int i = 0; i < comm_size; i++){
        MPI_Wait(&send_requests[i],MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < comm_size; i++){
        MPI_Wait(&recv_requests[i],MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&strided_t);

}

template<class T>
void Distribution<T>::reorder(T* Buff1, T* Buff2, int n, int direction, int batch, gpuStream_t stream){
    int dim = (n+2)%3;

    int nsends = (nlocal / world_size) / batches;

    T* Buff1Start = &Buff1[batch * (nlocal / batches)];
    T* Buff2Start = &Buff2[batch * (nlocal / batches)];

    if (direction == 0){
        launch_reorder_forward(Buff2Start,Buff1Start,nsends / local_grid_size[dim],world_size,local_grid_size[dim],blockSize,stream);
    } else {
        launch_reorder_backward(Buff2Start,Buff1Start,nsends / local_grid_size[dim],world_size,local_grid_size[dim],blockSize,stream);
    }
}

template<class T>
void Distribution<T>::memcpy_d2h(T* h, T* d, int batch, gpuStream_t stream){
    T* d_start = &d[batch * (nlocal / batches)];
    T* h_start = &h[batch * (nlocal / batches)];
    gpuMemcpyAsync(h_start,d_start,sizeof(T) * (nlocal / batches),gpuMemcpyDeviceToHost,stream);
}

template<class T>
void Distribution<T>::memcpy_h2d(T* d, T* h, int batch, gpuStream_t stream){
    T* d_start = &d[batch * (nlocal / batches)];
    T* h_start = &h[batch * (nlocal / batches)];
    gpuMemcpyAsync(d_start,h_start,sizeof(T) * (nlocal / batches),gpuMemcpyHostToDevice,stream);
}


template<class T>
void Distribution<T>::getPencils(T* Buff1, T* Buff2, int n, int batch){
    int dim = (n+2)%3;

    MPI_Comm my_comm = fftcomms[dim];

    int nsends = (nlocal / world_size) / batches;
    
    T* Buff1Start = &Buff1[batch * (nlocal / batches)];
    T* Buff2Start = &Buff2[batch * (nlocal / batches)];

    #ifdef customalltoall
    s_alltoall_forward(Buff1Start,Buff2Start,nsends,local_grid_size[dim],my_comm);
    #else
    MPI_Alltoall(Buff1Start,nsends,TYPE_COMPLEX,Buff2Start,nsends,TYPE_COMPLEX,my_comm);
    #endif
}

template<class T>
void Distribution<T>::returnPencils(T* Buff1, T* Buff2, int n, int batch){

    int dim = (n+2)%3;

    MPI_Comm my_comm = fftcomms[dim];

    int nsends = (nlocal / world_size) / batches;

    T* Buff1Start = &Buff1[batch * (nlocal / batches)];
    T* Buff2Start = &Buff2[batch * (nlocal / batches)];

    #ifdef customalltoall
    s_alltoall_backward(Buff1Start,Buff2Start,nsends,local_grid_size[dim],my_comm);
    #else
    MPI_Alltoall(Buff1Start,nsends,TYPE_COMPLEX,Buff2Start,nsends,TYPE_COMPLEX,my_comm);
    #endif
}

template<class T>
void Distribution<T>::finalize(){
    MPI_Type_free(&TYPE_COMPLEX);
    
    #ifdef nocudampi
    free(h_scratch1);
    free(h_scratch2);
    #endif

    #ifdef GPU
    cudaStreamDestroy(diststream);
    #endif
}

template<class T>
Distribution<T>::~Distribution(){
    finalize();
}


}

#endif