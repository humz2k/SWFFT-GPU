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
            
            #ifdef GPU
            void getPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int dim);
            void getPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int dim);
            #endif

            void getPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int dim);
            void getPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

            template<class T>
            inline void returnPencils_(T* Buff1, T* Buff2, int dim);

            #ifdef GPU
            void returnPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int dim);
            void returnPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int dim);
            #endif

            void returnPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int dim);
            void returnPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

            #ifdef GPU
            void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n);
            void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n);
            #endif

            void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n);
            void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n);

            #ifdef GPU
            void reorder(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n, int direction);
            void reorder(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n, int direction);
            #endif

            void reorder(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n, int direction);
            void reorder(complexFloatHost* Buff1, complexFloatHost* Buff2, int n, int direction);

    };

    template<class MPI_T, class REORDER_T, class FFTBackend>
    class Dfft {
        private:
            template<class T>
            inline void fft(T* data, T* scratch, fftdirection direction);
        
        public:
            int ng[3];
            int nlocal;
            int world_size;
            int world_rank;
            int blockSize;
            Distribution<MPI_T,REORDER_T>& dist;

            FFTBackend FFTs;

            Dfft(Distribution<MPI_T,REORDER_T>& dist_);
            ~Dfft();

            #ifdef GPU
            void forward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);
            void forward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
            #endif

            void forward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);
            void forward(complexFloatHost* Buff1, complexFloatHost* Buff2);
            
            #ifdef GPU
            void backward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);
            void backward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
            #endif

            void backward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);
            void backward(complexFloatHost* Buff1, complexFloatHost* Buff2);

    };

}
#ifdef GPU
template<class MPI_T,class FFTBackend>
class AllToAllGPU{
    private:
        A2A::Distribution<MPI_T,A2A::GPUReorder> dist;
        A2A::Dfft<MPI_T,A2A::GPUReorder,FFTBackend> dfft;

    public:
        AllToAllGPU(){

        }

        AllToAllGPU(MPI_Comm comm, int ngx, int blockSize) : dist(comm,ngx,ngx,ngx,blockSize), dfft(dist){

        }

        AllToAllGPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize) : dist(comm,ngx,ngx,ngx,blockSize), dfft(dist){

        }

        ~AllToAllGPU(){};

        int buff_sz(){
            return dist.nlocal;
        }

        int3 coords(){
            return make_int3(dist.coords[0],dist.coords[1],dist.coords[2]);
        }

        int rank(){
            return dist.world_rank;
        }

        MPI_Comm comm(){
            return dist.comm;
        }

        void forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            dfft.forward(data,scratch);
        }

        void forward(complexFloatDevice* data, complexFloatDevice* scratch){
            dfft.forward(data,scratch);
        }

        void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        void forward(complexFloatHost* data, complexFloatHost* scratch){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        void backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            dfft.backward(data,scratch);
        }

        void backward(complexFloatDevice* data, complexFloatDevice* scratch){
            dfft.backward(data,scratch);
        }

        void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        void backward(complexFloatHost* data, complexFloatHost* scratch){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        void forward(complexDoubleDevice* data){
            complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        void forward(complexFloatDevice* data){
            complexFloatDevice* scratch; swfftAlloc(&scratch,sizeof(complexFloatDevice) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        void forward(complexDoubleHost* data){
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        void forward(complexFloatHost* data){
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        void backward(complexDoubleDevice* data){
            complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        void backward(complexFloatDevice* data){
            complexFloatDevice* scratch; swfftAlloc(&scratch,sizeof(complexFloatDevice) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        void backward(complexDoubleHost* data){
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        void backward(complexFloatHost* data){
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }
};
#endif

template<class MPI_T,class FFTBackend>
class AllToAllCPU{
    private:
        A2A::Distribution<MPI_T,A2A::CPUReorder> dist;
        A2A::Dfft<MPI_T,A2A::CPUReorder,FFTBackend> dfft;

    public:
        AllToAllCPU(){

        }

        AllToAllCPU(MPI_Comm comm, int ngx, int blockSize) : dist(comm,ngx,ngx,ngx,blockSize), dfft(dist){

        }

        AllToAllCPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize) : dist(comm,ngx,ngx,ngx,blockSize), dfft(dist){

        }

        ~AllToAllCPU(){};

        int buff_sz(){
            return dist.nlocal;
        }

        int3 coords(){
            return make_int3(dist.coords[0],dist.coords[1],dist.coords[2]);
        }

        int rank(){
            return dist.world_rank;
        }

        MPI_Comm comm(){
            return dist.comm;
        }

        void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            dfft.forward(data,scratch);
        }

        void forward(complexFloatHost* data, complexFloatHost* scratch){
            dfft.forward(data,scratch);
        }

        #ifdef GPU
        void forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        void forward(complexFloatDevice* data, complexFloatDevice* scratch){
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }
        #endif

        void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            dfft.backward(data,scratch);
        }

        void backward(complexFloatHost* data, complexFloatHost* scratch){
            dfft.backward(data,scratch);
        }

        #ifdef GPU
        void backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        void backward(complexFloatDevice* data, complexFloatDevice* scratch){
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }
        #endif

        void forward(complexDoubleHost* data){
            complexDoubleHost* scratch; swfftAlloc(&scratch,sizeof(complexDoubleHost) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        void forward(complexFloatHost* data){
            complexFloatHost* scratch; swfftAlloc(&scratch,sizeof(complexFloatHost) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        #ifdef GPU
        void forward(complexDoubleDevice* data){
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        void forward(complexFloatDevice* data){
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }
        #endif

        void backward(complexDoubleHost* data){
            complexDoubleHost* scratch; swfftAlloc(&scratch,sizeof(complexDoubleHost) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        void backward(complexFloatHost* data){
            complexFloatHost* scratch; swfftAlloc(&scratch,sizeof(complexFloatHost) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        #ifdef GPU
        void backward(complexDoubleDevice* data){
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        void backward(complexFloatDevice* data){
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }
        #endif
};

#endif
#endif