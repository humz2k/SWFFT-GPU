#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
#ifdef SWFFT_GPUDELEGATE

#include "gpudelegate.hpp"

namespace SWFFT{

namespace GPUDELEGATE{

template<class MPI_T, class FFTBackend>
Dfft<MPI_T,FFTBackend>::Dfft(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_, int nsends_, int delegate_) : comm(comm_), ng{ngx,ngy,ngz}, blockSize(blockSize_), dims{0,0,0}, delegate(delegate_), nsends(nsends_), set(0){

    MPI_Comm_rank(comm,&world_rank);
    MPI_Comm_size(comm,&world_size);
    
    MPI_Dims_create(world_size,3,dims);

    int3_dims = make_int3(dims[0],dims[1],dims[2]);
    int3_ng = make_int3(ng[0],ng[1],ng[2]);

    local_grid_size[0] = ng[0] / dims[0];
    local_grid_size[1] = ng[1] / dims[1];
    local_grid_size[2] = ng[2] / dims[2];

    int3_local_grid_size = make_int3(local_grid_size[0],local_grid_size[1],local_grid_size[2]);

    nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    coords[0] = world_rank / (dims[1] * dims[2]);
    coords[1] = (world_rank - coords[0] * (dims[1] * dims[2])) / dims[2];
    coords[2] = (world_rank - coords[0] * (dims[1] * dims[2])) - coords[1] * dims[2];

    local_coords_start[0] = local_grid_size[0] * coords[0];
    local_coords_start[1] = local_grid_size[1] * coords[1];
    local_coords_start[2] = local_grid_size[2] * coords[2];

    //if (delegate == world_rank){
    //printf("Planning with %d %d %d\n",ng[0],ng[1],ng[2]);
    if(gpufftPlan3d(&planDouble,ng[0],ng[1],ng[2],GPUFFT_Z2Z) != GPUFFT_SUCCESS){
        printf("plan3d Double failed!\n");
    }
    if(gpufftPlan3d(&planFloat,ng[0],ng[1],ng[2],GPUFFT_C2C) != GPUFFT_SUCCESS){
        printf("plan3d Float failed!\n");
    }
    //}

}

template<class MPI_T, class FFTBackend>
Dfft<MPI_T,FFTBackend>::~Dfft(){

    //if(delegate == world_rank){
    if(gpufftDestroy(planDouble) != GPUFFT_SUCCESS){
        printf("destroy plan3d Double failed!\n");
    }
    if (gpufftDestroy(planFloat) != GPUFFT_SUCCESS){
        printf("destory plan3d Double failed\n");
    }
    //}

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::set_last_t(complexDoubleDevice* s){

    last_t = 0;

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::set_last_t(complexFloatDevice* s){

    last_t = 1;

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::execFFT(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int direction){
    //printf("exec3d!\n");
    if (gpufftExecZ2Z(planDouble,buff1,buff2,direction) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z failed\n");
        return;
    }

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::execFFT(complexFloatDevice* buff1, complexFloatDevice* buff2, int direction){

    if (gpufftExecC2C(planFloat,buff1,buff2,direction) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C failed\n");
        return;
    }

}

template<class MPI_T, class FFTBackend>
template<class T>
void Dfft<MPI_T,FFTBackend>::_fft(T* buff1, int direction){
    
    
    //int nsends = 4;
    int npersend = nlocal / nsends;

    set_last_t(buff1);

    if (world_rank == delegate){
        //printf("nsends = %d\n",nsends);
        //Irecv<MPI_T,T> recvs[world_size * nsends];
        //Isend<MPI_T,T> sends[world_size * nsends];
        if(set == 1){
            free(_recvs);
            free(_sends);
            gpuFree(_fftBuff1);
            gpuFree(_fftBuff2);
        }
        _recvs = malloc(sizeof(Irecv<MPI_T,T>) * world_size * nsends);
        _sends = malloc(sizeof(Isend<MPI_T,T>) * world_size * nsends);

        Irecv<MPI_T,T>* recvs = (Irecv<MPI_T,T>*)_recvs;
        Isend<MPI_T,T>* sends = (Isend<MPI_T,T>*)_sends;

        set = 1;

        T* fftBuff1; swfftAlloc(&fftBuff1,sizeof(T) * ng[0] * ng[1] * ng[2]);
        T* fftBuff2; swfftAlloc(&fftBuff2,sizeof(T) * ng[0] * ng[1] * ng[2]);
        _fftBuff1 = fftBuff1;
        _fftBuff2 = fftBuff2;
        
        reorder(0,buff1,fftBuff2,int3_ng,int3_local_grid_size,int3_dims,world_rank,0,blockSize);

        for (int i = 0; i < world_size; i++){
            if (i == world_rank)continue;
            for (int send = 0; send < nsends; send++){
                recvs[i*nsends + send] = mpi.irecv(&fftBuff1[nlocal * i + send * npersend],npersend,i,send,comm);
            }
        }

        for (int i = 0; i < world_size; i++){
            if (i == world_rank)continue;
            for (int send = 0; send < nsends; send++){
                recvs[i*nsends + send].execute();
            }
        }

        for (int i = 0; i < world_size; i++){
            if (i == world_rank)continue;
            for (int send = 0; send < nsends; send++){
                recvs[i*nsends + send].wait();
            }
        }

        for (int i = 0; i < world_size; i++){
            if (i == world_rank)continue;
            for (int send = 0; send < nsends; send++){
                recvs[i*nsends + send].finalize();
            }
            reorder(0,&fftBuff1[nlocal * i],fftBuff2,int3_ng,int3_local_grid_size,int3_dims,i,0,blockSize);
        }

        execFFT(fftBuff2,fftBuff1,direction);

        for (int i = 0; i < world_size; i++){
            if (i == world_rank)continue;
            reorder(1,fftBuff1,&fftBuff2[nlocal * i],int3_ng,int3_local_grid_size,int3_dims,i,0,blockSize);
            for (int send = 0; send < nsends; send++){
                sends[i*nsends + send] = mpi.isend(&fftBuff2[nlocal * i + send*npersend],npersend,i,send,comm);
            }
        }

        reorder(1,fftBuff1,buff1,int3_ng,int3_local_grid_size,int3_dims,world_rank,0,blockSize);

        for (int i = 0; i < world_size; i++){
            if (i == world_rank)continue;
            for (int send = 0; send < nsends; send++){
                sends[i*nsends + send].execute();
            }
        }

        

        //swfftFree(fftBuff1);
        //swfftFree(fftBuff2);

        //free(_recvs);
        //free(_sends);
        //set = 0;
    } else {
        //Isend<MPI_T,T> sends[nsends];
        //Irecv<MPI_T,T> recvs[nsends];
        if (set){
            free(_recvs);
            free(_sends);
        }
        _sends = malloc(sizeof(Isend<MPI_T,T>) * nsends);
        _recvs = malloc(sizeof(Irecv<MPI_T,T>) * nsends);
        Isend<MPI_T,T>* sends = (Isend<MPI_T,T>*)_sends;
        Irecv<MPI_T,T>* recvs = (Irecv<MPI_T,T>*)_recvs;
        set = 1;

        for (int send = 0; send < nsends; send++){
            sends[send] = mpi.isend(&buff1[send * npersend],npersend,delegate,send,comm);
            recvs[send] = mpi.irecv(&buff1[send * npersend],npersend,delegate,send,comm);
        }
        for (int send = 0; send < nsends; send++){
            sends[send].execute();
        }
        for (int send = 0; send < nsends; send++){
            recvs[send].execute();
        }
        

        //free(_recvs);
        //free(_sends);
        //set = 0;
    }

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::synchronize(){
    if(!set){
        //printf("???");
        return;
    }
    if(last_t == 0){
        Isend<MPI_T,complexDoubleDevice>* sends = (Isend<MPI_T,complexDoubleDevice>*)_sends;
        Irecv<MPI_T,complexDoubleDevice>* recvs = (Irecv<MPI_T,complexDoubleDevice>*)_recvs;
        if (world_rank == delegate){
            for (int i = 0; i < world_size; i++){
                if (i == world_rank)continue;
                for (int send = 0; send < nsends; send++){
                    sends[i*nsends + send].wait();
                }
            }

        } else {
            for (int send = 0; send < nsends; send++){
                sends[send].wait();
            }
            for (int send = 0; send < nsends; send++){
                recvs[send].wait();
            }
            for (int send = 0; send < nsends; send++){
                recvs[send].finalize();
            }
        }
    }

    if(last_t == 1){
        Isend<MPI_T,complexFloatDevice>* sends = (Isend<MPI_T,complexFloatDevice>*)_sends;
        Irecv<MPI_T,complexFloatDevice>* recvs = (Irecv<MPI_T,complexFloatDevice>*)_recvs;
        if (world_rank == delegate){
            for (int i = 0; i < world_size; i++){
                if (i == world_rank)continue;
                for (int send = 0; send < nsends; send++){
                    sends[i*nsends + send].wait();
                }
            }

        } else {
            for (int send = 0; send < nsends; send++){
                sends[send].wait();
            }
            for (int send = 0; send < nsends; send++){
                recvs[send].wait();
            }
            for (int send = 0; send < nsends; send++){
                recvs[send].finalize();
            }
        }
    }

    gpuStreamSynchronize(0);

    if (world_rank == delegate){
        gpuFree(_fftBuff1);
        gpuFree(_fftBuff2);
    }

    free(_sends);
    free(_recvs);
    set = 0;

}

template<class MPI_T, class FFTBackend>
template<class T>
void Dfft<MPI_T,FFTBackend>::_fftNoReorder(T* buff1, int direction){
    
    int sends_per_rank = local_grid_size[0] * local_grid_size[1];

    if (world_rank == delegate){
        //printf("rank %d is delegate\n",world_rank);

        Irecv<MPI_T,T> recvs[world_size * sends_per_rank];
        Isend<MPI_T,T> sends[world_size * sends_per_rank];

        T* fftBuff1; swfftAlloc(&fftBuff1,sizeof(T) * ng[0] * ng[1] * ng[2]);
        
        reorder(0,buff1,fftBuff1,int3_ng,int3_local_grid_size,int3_dims,world_rank,0,blockSize);

        int count = 0;
        for (int i = 0; i < world_size; i++){

            int3 rank_coords;
            rank_coords.x = i / (dims[1] * dims[2]);
            rank_coords.y = (i - (rank_coords.x * dims[1] * dims[2])) / dims[2];
            rank_coords.z = (i - (rank_coords.x * dims[1] * dims[2])) - rank_coords.y * dims[2];

            int3 rank_start;
            rank_start.x = rank_coords.x * local_grid_size[0];
            rank_start.y = rank_coords.y * local_grid_size[1];
            rank_start.z = rank_coords.z * local_grid_size[2];

            int local_count = 0;
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    int3 this_start;
                    this_start.x = rank_start.x + x;
                    this_start.y = rank_start.y + y;
                    this_start.z = rank_start.z;

                    int start_idx = this_start.x * ng[1] * ng[2] + this_start.y * ng[2] + this_start.z;

                    if(i != world_rank){
                        printf("init recv %d from %d\n",local_count,i);
                        recvs[count] = mpi.irecv(&fftBuff1[start_idx],local_grid_size[2],i,local_count,comm);
                    }
                    count++;
                    local_count++;

                }
            }
        }

        count = 0;
        for (int i = 0; i < world_size; i++){
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    if (i != world_rank){
                        printf("execute recv from %d\n",i);
                        recvs[count].execute();
                    }
                    count++;
                }
            }
        }

        count = 0;
        for (int i = 0; i < world_size; i++){
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    if (i != world_rank){
                        printf("waiting recv from %d\n",i);
                        recvs[count].wait();
                    }
                    count++;
                }
            }
        }

        count = 0;
        for (int i = 0; i < world_size; i++){
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    if (i != world_rank){
                        printf("finalize recv from %d\n",i);
                        recvs[count].finalize();
                    }
                    count++;
                }
            }
        }

        printf("execute fft\n");

        execFFT(fftBuff1,fftBuff1,direction);

        count = 0;
        for (int i = 0; i < world_size; i++){

            int3 rank_coords;
            rank_coords.x = i / (dims[1] * dims[2]);
            rank_coords.y = (i - (rank_coords.x * dims[1] * dims[2])) / dims[2];
            rank_coords.z = (i - (rank_coords.x * dims[1] * dims[2])) - rank_coords.y * dims[2];

            int3 rank_start;
            rank_start.x = rank_coords.x * local_grid_size[0];
            rank_start.y = rank_coords.y * local_grid_size[1];
            rank_start.z = rank_coords.z * local_grid_size[2];

            int local_count = 0;
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    int3 this_start;
                    this_start.x = rank_start.x + x;
                    this_start.y = rank_start.y + y;
                    this_start.z = rank_start.z;

                    int start_idx = this_start.x * ng[1] * ng[2] + this_start.y * ng[2] + this_start.z;
                    if (i != world_rank){
                        printf("sending %d to %d\n",local_count,i);
                        sends[count] = mpi.isend(&fftBuff1[start_idx],local_grid_size[2],i,local_count,comm);
                    }
                    count++;
                    local_count++;

                }
            }
        }

        reorder(1,fftBuff1,buff1,int3_ng,int3_local_grid_size,int3_dims,world_rank,0,blockSize);

        count = 0;
        for (int i = 0; i < world_size; i++){
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    if (i != world_rank){
                        printf("executing send to %d\n",i);
                        sends[count].execute();
                    }
                    count++;
                }
            }
        }

        count = 0;
        for (int i = 0; i < world_size; i++){
            for (int x = 0; x < local_grid_size[0]; x++){
                for (int y = 0; y < local_grid_size[1]; y++){
                    if (i != world_rank){
                        printf("waiting send to %d\n",i);
                        sends[count].wait();
                    }
                    count++;
                }
            }
        }

        gpuStreamSynchronize(0);

        swfftFree(fftBuff1);
    } else {
        Isend<MPI_T,T> sends[sends_per_rank];
        for (int i = 0; i < sends_per_rank; i++){
            sends[i] = mpi.isend(&buff1[i * local_grid_size[2]],local_grid_size[2],delegate,i,comm);
        }
        for (int i = 0; i < sends_per_rank; i++){
            sends[i].execute();
        }
        for (int i = 0; i < sends_per_rank; i++){
            sends[i].wait();
        }

        Irecv<MPI_T,T> recvs[sends_per_rank];
        for (int i = 0; i < sends_per_rank; i++){
            recvs[i] = mpi.irecv(&buff1[i * local_grid_size[2]],local_grid_size[2],delegate,i,comm);
        }
        for (int i = 0; i < sends_per_rank; i++){
            recvs[i].execute();
        }
        for (int i = 0; i < sends_per_rank; i++){
            recvs[i].wait();
        }
        for (int i = 0; i < sends_per_rank; i++){
            recvs[i].finalize();
        }
    }

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::fft(complexDoubleDevice* buff1, int direction){
    
    _fft(buff1,direction);
    //synchronize();

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::fft(complexFloatDevice* buff1, int direction){
    
    _fft(buff1,direction);
    //synchronize();

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::forward(complexDoubleDevice* buff1){
    
    fft(buff1,GPUFFT_FORWARD);

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::forward(complexFloatDevice* buff1){
    
    fft(buff1,GPUFFT_FORWARD);

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::backward(complexDoubleDevice* buff1){
    
    fft(buff1,GPUFFT_INVERSE);

}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::backward(complexFloatDevice* buff1){
    
    fft(buff1,GPUFFT_INVERSE);

}

template<class MPI_T, class FFTBackend>
int Dfft<MPI_T,FFTBackend>::buff_sz(){
    
    return nlocal;

}

template class Dfft<CPUMPI,gpuFFT>;

}
}

#endif
#endif
#endif