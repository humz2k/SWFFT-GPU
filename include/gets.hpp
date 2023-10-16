#ifndef _GET_MAP_
#define _GET_MAP_

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

struct map_return_t{
    int rank;
    int src_idx;
    int dest_idx;
};

inline map_return_t make_map_return(int rank, int src_idx, int dest_idx){
    map_return_t out;
    out.rank = rank;
    out.src_idx = src_idx;
    out.dest_idx = dest_idx;
    return out;
}

struct get_t{
    int rank;
    int src;
    int dest;
    int stride;
    int n;
    get_t* next;
};

struct get_serial_t{
    int src;
    int dest;
    int stride;
    int n;
};

inline get_t* make_get(int rank, int src, int dest, int stride, int n){
    get_t* out = (get_t*)malloc(sizeof(get_t));
    out->rank = rank;
    out->src = src;
    out->dest = dest;
    out->stride = stride;
    out->n = n;
    out->next = NULL;
    return out;
}

inline get_t* make_empty_get(){
    get_t* out = (get_t*)malloc(sizeof(get_t));
    out->rank = -1;
    out->next = NULL;
    return out;
}

inline void free_get(get_t* out){
    get_t* tmp = out;
    while (tmp->next){
        get_t* prev = tmp;
        tmp = tmp->next;
        free(prev);
    }
    free(tmp);
}

inline void append_get(get_t* base, get_t* add){
    get_t* tmp = base;
    while (tmp->next){
        tmp = tmp->next;
    }
    tmp->next = add;
}

inline get_t* add_get(get_t* base, map_return_t ret){
    if ((base)->rank == ret.rank){
        if ((base)->n == 1){
            if ((base)->src < ret.src_idx){
                (base)->stride = ret.src_idx - (base)->src;
                (base)->n++;
                return base;
            }
        }
        if (((base)->n * (base)->stride + (base)->src) == ret.src_idx){
            (base)->n++;
            return base;
        }
    }
    
    if(base->rank != -1){
        get_t* new_get = make_get(ret.rank,ret.src_idx,ret.dest_idx,1,1);
        append_get(base,new_get);
        return new_get;
    }
    base->rank = ret.rank;
    base->n = 1;
    base->next = NULL;
    base->dest = ret.dest_idx;
    base->stride = 1;
    base->src = ret.src_idx;
    return base;
}

template<class Map>
inline get_t* find_gets(int n, Map& map){
    get_t* init = make_empty_get();
    get_t* cur = init;
    for (int i = 0; i < n; i++){
        map_return_t ret = map.map(i);
        cur = add_get(cur,ret);
    }
    return init;
}

inline int count_ranks(get_t* in, int* nsends, int* nelems, int* get_buff_starts, int nranks){
    for (int i = 0; i < nranks; i++){
        nsends[i] = 0;
        nelems[i] = 0;
    }
    int total_sends = 0;
    get_t* cur = in;
    while (cur){
        nsends[cur->rank]++;
        nelems[cur->rank] += cur->n;
        cur = cur->next;
        total_sends++;
    }

    get_buff_starts[0] = 0;
    for (int i = 1; i < nranks; i++){
        get_buff_starts[i] = get_buff_starts[i-1] + nelems[i-1];
    }
    return total_sends;

}

inline get_serial_t* unify(get_t* in, int* nsends, int* starts, int total_sends, int nranks){

    

    get_serial_t* out = (get_serial_t*)malloc(sizeof(get_serial_t)*total_sends);
    starts[0] = 0;
    for (int i = 1; i < nranks; i++){
        starts[i] = starts[i-1] + nsends[i-1];
    }

    int counts[nranks];
    for (int i = 0; i < nranks; i++){
        counts[i] = 0;
    }

    
    get_t* cur = in;
    while (cur){
        int rank = cur->rank;
        int off = starts[rank] + counts[rank]++;
        out[off].src = cur->src;
        out[off].dest = cur->dest;
        out[off].n = cur->n;
        out[off].stride = cur->stride;
        cur = cur->next;
    }

    free_get(in);

    return out;
}

inline get_serial_t* distribute(get_serial_t* in, int* ngets, int* starts, int* nsends, int* send_starts, int* total_sends, int total_gets, int nranks, MPI_Comm comm){
    int tmp_total_sends = 0;
    MPI_Alltoall(ngets,1,MPI_INT,nsends,1,MPI_INT,comm);
    for (int i = 0; i < nranks; i++){
        tmp_total_sends += nsends[i];
    }

    send_starts[0] = 0;
    for (int i = 1; i < nranks; i++){
        send_starts[i] = send_starts[i-1] + nsends[i-1];
    }

    *total_sends = tmp_total_sends;
    get_serial_t* out = (get_serial_t*)malloc(sizeof(get_serial_t) * tmp_total_sends);

    MPI_Datatype tmp_type;
    MPI_Type_contiguous(sizeof(get_serial_t),MPI_BYTE,&tmp_type);
    MPI_Type_commit(&tmp_type);

    MPI_Alltoallv(in,ngets,starts,tmp_type,out,nsends,send_starts,tmp_type,comm);

    MPI_Type_free(&tmp_type);

    return out;
}

inline void count_sends(get_serial_t* in, int* nsends, int* send_starts, int* rank_send_starts, int* rank_send_ns, int nranks){
    for (int i = 0; i < nranks; i++){
        rank_send_ns[i] = 0;
        rank_send_starts[i] = 0;
        if (nsends[i] == 0)continue;
        for (int j = 0; j < nsends[j]; j++){
            rank_send_ns[i] += in[send_starts[i] + j].n;
        }
    }
    for (int i = 1; i < nranks; i++){
        rank_send_starts[i] = rank_send_starts[i-1] + rank_send_ns[i-1];
    }
}

#endif