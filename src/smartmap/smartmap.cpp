#ifdef SWFFT_SMARTMAP

#include "smartmap.hpp"

namespace SWFFT{

namespace SMARTMAP{

template<class MPI_T, class FFTBackend>
Dfft<MPI_T,FFTBackend>::Dfft(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_) : comm(comm_), ng{ngy,ngy,ngz}, dims{0,0,0}, coords{0,0,0}, local_grid_size{0,0,0}{

    MPI_Comm_rank(comm,&comm_rank);
    MPI_Comm_size(comm,&comm_size);
    
    MPI_Dims_create(comm_size,3,dims);

    local_grid_size[0] = ng[0] / dims[0];
    local_grid_size[1] = ng[1] / dims[1];
    local_grid_size[2] = ng[2] / dims[2];

    nlocal = local_grid_size[0] * local_grid_size[1] * local_grid_size[2];

    coords[0] = comm_rank / (dims[1] * dims[2]);
    coords[1] = (comm_rank - coords[0] * (dims[1] * dims[2])) / dims[2];
    coords[2] = (comm_rank - coords[0] * (dims[1] * dims[2])) - coords[1] * dims[2];

    local_coords_start[0] = local_grid_size[0] * coords[0];
    local_coords_start[1] = local_grid_size[1] * coords[1];
    local_coords_start[2] = local_grid_size[2] * coords[2];

    int3 ng_vec = make_int3(ng[0],ng[1],ng[2]);
    int3 local_grid_size_vec = make_int3(local_grid_size[0],local_grid_size[1],local_grid_size[2]);
    int3 dims_vec = make_int3(dims[0],dims[1],dims[2]);
    int3 coords_vec = make_int3(coords[0],coords[1],coords[2]);

    map1 = map_1(comm,ng_vec,local_grid_size_vec,dims_vec,coords_vec,nlocal);
    map2 = map_2(comm,ng_vec,local_grid_size_vec,dims_vec,coords_vec,nlocal);
    map3 = map_3(comm,ng_vec,local_grid_size_vec,dims_vec,coords_vec,nlocal);

    sm1 = new SmartMap<map_1>(comm,map1,nlocal);
    sm2 = new SmartMap<map_2>(comm,map2,nlocal);
    sm3 = new SmartMap<map_3>(comm,map3,nlocal);

}

template<class MPI_T, class FFTBackend>
Dfft<MPI_T,FFTBackend>::~Dfft(){

    delete sm1;
    delete sm2;
    delete sm3;

}

template<class MPI_T, class FFTBackend>
template<class T>
void Dfft<MPI_T,FFTBackend>::_forward(T* data, T* scratch){
    sm1->forward(data,scratch);
    ffts.forward(scratch,data,ng[2],nlocal/ng[2]);
    sm2->forward(data,scratch);
    ffts.forward(scratch,data,ng[1],nlocal/ng[1]);
    sm3->forward(data,scratch);
    ffts.forward(scratch,data,ng[0],nlocal/ng[0]);
}

template<class MPI_T, class FFTBackend>
template<class T>
void Dfft<MPI_T,FFTBackend>::_backward(T* data, T* scratch){
    
    ffts.backward(data,scratch,ng[0],nlocal/ng[0]);
    sm3->backward(scratch,data);
    
    ffts.backward(data,scratch,ng[1],nlocal/ng[1]);
    sm2->backward(scratch,data);
    
    ffts.backward(data,scratch,ng[2],nlocal/ng[2]);
    sm1->backward(scratch,data);
}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::forward(complexDoubleHost* data, complexDoubleHost* scratch){
    _forward(data,scratch);
}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::backward(complexDoubleHost* data, complexDoubleHost* scratch){
    _backward(data,scratch);
}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::forward(complexFloatHost* data, complexFloatHost* scratch){
    _forward(data,scratch);
}

template<class MPI_T, class FFTBackend>
void Dfft<MPI_T,FFTBackend>::backward(complexFloatHost* data, complexFloatHost* scratch){
    _backward(data,scratch);
}

template<class MPI_T, class FFTBackend>
int Dfft<MPI_T,FFTBackend>::buff_sz(){
    return nlocal;
}

template class Dfft<CPUMPI,TestFFT>;
#ifdef SWFFT_FFTW
template class Dfft<CPUMPI,fftw>;
#endif
#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
template class Dfft<CPUMPI,gpuFFT>;
#endif
#ifndef SWFFT_NOCUDAMPI
template class Dfft<GPUMPI,gpuFFT>;
template class Dfft<GPUMPI,fftw>;
template class Dfft<GPUMPI,TestFFT>;
#endif
#endif

}

}

#endif