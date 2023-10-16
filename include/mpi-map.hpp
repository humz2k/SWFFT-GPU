#ifndef _MAP_MAP_
#define _MAP_MAP_

#include <mpi.h>
#include "gets.hpp"

template<class Map>
class SmartMap{
    private:
        Map map;
        int comm_rank;
        int comm_size;
        MPI_Comm comm;
        int n;
        int* ngets;
        int* nelems;
        int* starts;
        int* nsends;
        int* send_starts;
        int* rank_buff_starts;
        int* rank_buff_ns;
        int* get_buff_starts;

        get_serial_t* gets;
        get_serial_t* sends;

        int total_sends;
        int total_gets;
        bool init;

    public:
        inline SmartMap() : init(false){

        }

        inline SmartMap(MPI_Comm comm_, Map map_, int n_) : comm(comm_), map(map_), n(n_), gets(NULL), sends(NULL), init(true){
            MPI_Comm_rank(comm,&comm_rank);
            MPI_Comm_size(comm,&comm_size);

            ngets = (int*)malloc(sizeof(int) * comm_size);
            nelems = (int*)malloc(sizeof(int) * comm_size);
            starts = (int*)malloc(sizeof(int) * comm_size);
            nsends = (int*)malloc(sizeof(int) * comm_size);
            send_starts = (int*)malloc(sizeof(int) * comm_size);
            rank_buff_starts = (int*)malloc(sizeof(int) * comm_size);
            rank_buff_ns = (int*)malloc(sizeof(int) * comm_size);
            get_buff_starts = (int*)malloc(sizeof(int) * comm_size);

            get_t* init = find_gets(n,map);

            total_gets = count_ranks(init,ngets,nelems,get_buff_starts,comm_size);

            gets = unify(init,ngets,starts,total_gets,comm_size);

            sends = distribute(gets,ngets,starts,nsends,send_starts,&total_sends,total_gets,comm_size,comm);

            count_sends(sends,nsends,send_starts,rank_buff_starts,rank_buff_ns,comm_size);

        }

        template<class T>
        inline void forward(T* in, T* out){
            if (!init)return;
            MPI_Request send_reqs[comm_size];
            MPI_Request get_reqs[comm_size];

            int count = 0;
            for (int i = 0; i < comm_size; i++){

                if (nsends[i] == 0)continue;

                int start = count;
                for (int j = 0; j < nsends[i]; j++){
                    get_serial_t this_get = sends[send_starts[i] + j];
                    for (int k = 0; k < this_get.n; k++){
                        int get_idx = this_get.src + k*this_get.stride;
                        int out_idx = count++;
                        out[out_idx] = in[get_idx];
                    }
                }

                MPI_Isend(&out[start],(count - start) * sizeof(T),MPI_BYTE,i,0,comm,&send_reqs[i]);
            }

            count = 0;
            for (int i = 0; i < comm_size; i++){
                if (ngets[i] == 0)continue;
                MPI_Irecv(&in[count],nelems[i] * sizeof(T),MPI_BYTE,i,0,comm,&get_reqs[i]);
                count += nelems[i];
            }

            for (int i = 0; i < comm_size; i++){
                if (nsends[i] == 0)continue;
                MPI_Wait(&send_reqs[i],MPI_STATUS_IGNORE);
            }

            count = 0;
            for (int i = 0; i < comm_size; i++){
                if(ngets[i] == 0)continue;
                MPI_Wait(&get_reqs[i],MPI_STATUS_IGNORE);
                for (int j = 0; j < ngets[i]; j++){
                    get_serial_t this_get = gets[starts[i] + j];
                    for (int k = 0; k < this_get.n; k++){
                        int out_idx = this_get.dest + k;
                        int get_idx = count++;
                        out[out_idx] = in[get_idx];
                    }
                }
            }

        }

        template<class T>
        inline void backward(T* in, T* out){
            if (!init)return;
            MPI_Request send_reqs[comm_size];
            MPI_Request get_reqs[comm_size];

            int count = 0;
            for (int i = 0; i < comm_size; i++){
                if(ngets[i] == 0)continue;
                for (int j = 0; j < ngets[i]; j++){
                    get_serial_t this_get = gets[starts[i] + j];
                    for (int k = 0; k < this_get.n; k++){
                        int out_idx = this_get.dest + k;
                        int get_idx = count++;
                        out[get_idx] = in[out_idx];
                    }
                }
            }

            count = 0;
            for (int i = 0; i < comm_size; i++){
                if (ngets[i] == 0)continue;
                MPI_Isend(&out[count],nelems[i] * sizeof(T),MPI_BYTE,i,0,comm,&send_reqs[i]);
                count += nelems[i];
            }

            count = 0;
            for (int i = 0; i < comm_size; i++){

                if (nsends[i] == 0)continue;

                int start = count;
                for (int j = 0; j < nsends[i]; j++){
                    get_serial_t this_get = sends[send_starts[i] + j];
                    count += this_get.n;
                }

                MPI_Irecv(&in[start],(count - start) * sizeof(T),MPI_BYTE,i,0,comm,&get_reqs[i]);
                
            }

            for (int i = 0; i < comm_size; i++){
                if (ngets[i] == 0)continue;
                MPI_Wait(&send_reqs[i],MPI_STATUS_IGNORE);
            }

            for (int i = 0; i < comm_size; i++){
                if (nsends[i] == 0)continue;
                MPI_Wait(&get_reqs[i],MPI_STATUS_IGNORE);
            }

            count = 0;
            for (int i = 0; i < comm_size; i++){

                if (nsends[i] == 0)continue;

                int start = count;
                for (int j = 0; j < nsends[i]; j++){
                    get_serial_t this_get = sends[send_starts[i] + j];
                    for (int k = 0; k < this_get.n; k++){
                        int get_idx = this_get.src + k*this_get.stride;
                        int out_idx = count++;
                        out[get_idx] = in[out_idx];
                    }
                }
            }

        }

        inline ~SmartMap(){
            if (!init)return;
            free(ngets);
            free(nelems);
            free(starts);
            free(nsends);
            free(send_starts);
            free(rank_buff_ns);
            free(rank_buff_starts);
            free(get_buff_starts);
            if(gets){
                //printf("freeing gets\n");
                free(gets);
            }
            if(sends){
                //printf("freeing sends\n");
                free(sends);
            }
        }



};

#endif