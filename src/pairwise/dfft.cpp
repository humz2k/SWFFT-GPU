#ifdef SWFFT_PAIRWISE

#define PENCIL

#include "pairwise.hpp"
#include <cassert>

#ifndef USE_SLAB_WORKAROUND
#define USE_SLAB_WORKAROUND 0
#endif

#define DEBUG_CONDITION false
namespace SWFFT{
namespace PAIR{
    
    template<class MPI_T, class FFTBackend>
    Dfft<MPI_T,FFTBackend>::Dfft(MPI_Comm comm_, int nx, int ny, int nz) : comm(comm), n{nx,ny,nz}, double_dist(comm_,nx,ny,nz,DEBUG_CONDITION), float_dist(comm_,nx,ny,nz,DEBUG_CONDITION){

    }

    template<class MPI_T, class FFTBackend>
    int3 Dfft<MPI_T,FFTBackend>::coords(){
        return make_int3(double_dist.get_self_3d(0),double_dist.get_self_3d(1),double_dist.get_self_3d(2));
    }

    template<class MPI_T, class FFTBackend>
    int3 Dfft<MPI_T,FFTBackend>::get_ks(int idx){
        int3 local_ng_k = make_int3(double_dist.local_ng_2d_z(0),double_dist.local_ng_2d_z(1),double_dist.local_ng_2d_z(2));
        int3 pos_k = make_int3(double_dist.get_self_2d_z(0),double_dist.get_self_2d_z(1),double_dist.get_self_2d_z(2));
        int3 my_pos;
        my_pos.x = idx / (local_ng_k.y * local_ng_k.z);
        my_pos.y = (idx - (my_pos.x * local_ng_k.y * local_ng_k.z)) / local_ng_k.z;
        my_pos.z = (idx - (my_pos.x * local_ng_k.y * local_ng_k.z)) - my_pos.y * local_ng_k.z;
        my_pos.x += pos_k.x;
        my_pos.y += pos_k.y;
        my_pos.z += pos_k.z;
        return my_pos;
    }

    template<class MPI_T, class FFTBackend>
    Dfft<MPI_T,FFTBackend>::~Dfft(){}

    template<class MPI_T, class FFTBackend>
    template<class T>
    void Dfft<MPI_T,FFTBackend>::_forward(T* data){
        T* scratch; swfftAlloc(&scratch,sizeof(T) * buff_sz());
        forward(data,scratch);
        swfftFree(scratch);
    }

    template<class MPI_T, class FFTBackend>
    template<class T>
    void Dfft<MPI_T,FFTBackend>::_backward(T* data){
        T* scratch; swfftAlloc(&scratch,sizeof(T) * buff_sz());
        backward(data,scratch);
        swfftFree(scratch);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexDoubleHost* data){
        _forward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexFloatHost* data){
        _forward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexDoubleHost* data){
        _backward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexFloatHost* data){
        _backward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexDoubleHost* data, complexDoubleHost* scratch){
        double_dist.dist_3_to_2(data,scratch,0);

        FFTs.forward(scratch,data,double_dist.process_topology_2_x.n[0],double_dist.process_topology_2_x.n[1] * double_dist.process_topology_2_x.n[2]);

        double_dist.dist_2_to_3(data,scratch,0);

        double_dist.dist_3_to_2(scratch,data,1);

        FFTs.forward(data,scratch,double_dist.process_topology_2_y.n[1],double_dist.process_topology_2_y.n[0] * double_dist.process_topology_2_y.n[2]);

        double_dist.dist_2_to_3(scratch,data,1);

        double_dist.dist_3_to_2(data,scratch,2);

        FFTs.forward(scratch,data,double_dist.process_topology_2_z.n[2],double_dist.process_topology_2_z.n[0] * double_dist.process_topology_2_z.n[1]);

    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexFloatHost* data, complexFloatHost* scratch){
        float_dist.dist_3_to_2(data,scratch,0);

        FFTs.forward(scratch,data,float_dist.process_topology_2_x.n[0],float_dist.process_topology_2_x.n[1] * float_dist.process_topology_2_x.n[2]);

        float_dist.dist_2_to_3(data,scratch,0);

        float_dist.dist_3_to_2(scratch,data,1);

        FFTs.forward(data,scratch,float_dist.process_topology_2_y.n[1],float_dist.process_topology_2_y.n[0] * float_dist.process_topology_2_y.n[2]);

        float_dist.dist_2_to_3(scratch,data,1);

        float_dist.dist_3_to_2(data,scratch,2);

        FFTs.forward(scratch,data,float_dist.process_topology_2_z.n[2],float_dist.process_topology_2_z.n[0] * float_dist.process_topology_2_z.n[1]);

    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexDoubleHost* data, complexDoubleHost* scratch){

        FFTs.backward(data,scratch,double_dist.process_topology_2_z.n[2],double_dist.process_topology_2_z.n[0] * double_dist.process_topology_2_z.n[1]);

        double_dist.dist_2_to_3(scratch,data,2);

        double_dist.dist_3_to_2(data,scratch,1);

        FFTs.forward(scratch,data,double_dist.process_topology_2_y.n[1],double_dist.process_topology_2_y.n[0] * double_dist.process_topology_2_y.n[2]);

        double_dist.dist_2_to_3(data,scratch,1);

        double_dist.dist_3_to_2(scratch,data,0);

        FFTs.forward(data,scratch,double_dist.process_topology_2_x.n[0],double_dist.process_topology_2_x.n[1] * double_dist.process_topology_2_x.n[2]);

        double_dist.dist_2_to_3(scratch,data,0);

    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexFloatHost* data, complexFloatHost* scratch){

        FFTs.backward(data,scratch,float_dist.process_topology_2_z.n[2],float_dist.process_topology_2_z.n[0] * float_dist.process_topology_2_z.n[1]);

        float_dist.dist_2_to_3(scratch,data,2);

        float_dist.dist_3_to_2(data,scratch,1);

        FFTs.forward(scratch,data,float_dist.process_topology_2_y.n[1],float_dist.process_topology_2_y.n[0] * float_dist.process_topology_2_y.n[2]);

        float_dist.dist_2_to_3(data,scratch,1);

        float_dist.dist_3_to_2(scratch,data,0);

        FFTs.forward(data,scratch,float_dist.process_topology_2_x.n[0],float_dist.process_topology_2_x.n[1] * float_dist.process_topology_2_x.n[2]);

        float_dist.dist_2_to_3(scratch,data,0);

    }

    #ifdef SWFFT_GPU
    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexDoubleDevice* data){
        size_t sz = sizeof(complexDoubleDevice) * buff_sz();
        complexDoubleHost* h_data; swfftAlloc(&h_data,sz);
        gpuMemcpy(h_data,data,sz,gpuMemcpyDeviceToHost);
        forward(h_data);
        gpuMemcpy(data,h_data,sz,gpuMemcpyHostToDevice);
        swfftFree(h_data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexFloatDevice* data){
        size_t sz = sizeof(complexFloatDevice) * buff_sz();
        complexFloatHost* h_data; swfftAlloc(&h_data,sz);
        gpuMemcpy(h_data,data,sz,gpuMemcpyDeviceToHost);
        forward(h_data);
        gpuMemcpy(data,h_data,sz,gpuMemcpyHostToDevice);
        swfftFree(h_data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexDoubleDevice* data){
        size_t sz = sizeof(complexDoubleDevice) * buff_sz();
        complexDoubleHost* h_data; swfftAlloc(&h_data,sz);
        gpuMemcpy(h_data,data,sz,gpuMemcpyDeviceToHost);
        backward(h_data);
        gpuMemcpy(data,h_data,sz,gpuMemcpyHostToDevice);
        swfftFree(h_data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexFloatDevice* data){
        size_t sz = sizeof(complexFloatDevice) * buff_sz();
        complexFloatHost* h_data; swfftAlloc(&h_data,sz);
        gpuMemcpy(h_data,data,sz,gpuMemcpyDeviceToHost);
        backward(h_data);
        gpuMemcpy(data,h_data,sz,gpuMemcpyHostToDevice);
        swfftFree(h_data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
        forward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::forward(complexFloatDevice* data, complexFloatDevice* scratch){
        forward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
        backward(data);
    }

    template<class MPI_T, class FFTBackend>
    void Dfft<MPI_T,FFTBackend>::backward(complexFloatDevice* data, complexFloatDevice* scratch){
        backward(data);
    }

    #endif

    template<class MPI_T, class FFTBackend>
    int Dfft<MPI_T,FFTBackend>::get_nproc_3d(int direction){
        return double_dist.get_nproc_3d(direction);
    }

    template<class MPI_T, class FFTBackend>
    int Dfft<MPI_T,FFTBackend>::buff_sz(){
        int size = 1;
        for (int i = 0; i < 3; i++){
            size *= n[i] / double_dist.process_topology_3.nproc[i];
        }
        return size;
    }

    #ifdef SWFFT_FFTW
    template class Dfft<CPUMPI,fftw>;
    #endif
    #ifdef SWFFT_GPU
    #ifdef SWFFT_CUFFT
    template class Dfft<CPUMPI,gpuFFT>;
    #endif
    #endif

}
}

#endif