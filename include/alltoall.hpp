#ifdef SWFFT_ALLTOALL
#ifndef SWFFT_ALLTOALL_SEEN
#define SWFFT_ALLTOALL_SEEN

#include "alltoall_reorder.hpp"
#include <mpi.h>
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"

namespace SWFFT{

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
            bool ks_as_block;
            MPI_Comm comm;
            MPI_Comm fftcomms[3];

            MPI_T mpi;
            REORDER_T reordering;

            int blockSize;

            Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_, bool ks_as_block_);
            ~Distribution();

            MPI_Comm shuffle_comm_1();
            MPI_Comm shuffle_comm_2();
            MPI_Comm shuffle_comm(int n);

            template<class T>
            inline void getPencils_(T* Buff1, T* Buff2, int dim);

            #ifdef SWFFT_GPU
            void copy(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);
            void copy(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
            #endif

            void copy(complexDoubleHost* __restrict Buff1, const complexDoubleHost* __restrict Buff2);
            void copy(complexFloatHost* __restrict Buff1, const complexFloatHost* __restrict Buff2);
            
            #ifdef SWFFT_GPU
            void getPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int dim);
            void getPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int dim);
            #endif

            void getPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int dim);
            void getPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

            template<class T>
            inline void returnPencils_(T* Buff1, T* Buff2, int dim);

            #ifdef SWFFT_GPU
            void returnPencils(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int dim);
            void returnPencils(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int dim);
            #endif

            void returnPencils(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int dim);
            void returnPencils(complexFloatHost* Buff1, complexFloatHost* Buff2, int dim);

            #ifdef SWFFT_GPU
            void shuffle_indices(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2, int n);
            void shuffle_indices(complexFloatDevice* Buff1, complexFloatDevice* Buff2, int n);
            #endif

            void shuffle_indices(complexDoubleHost* Buff1, complexDoubleHost* Buff2, int n);
            void shuffle_indices(complexFloatHost* Buff1, complexFloatHost* Buff2, int n);

            #ifdef SWFFT_GPU
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
            inline double fft(T* data, T* scratch, fftdirection direction);
            bool ks_as_block;

            #ifdef SWFFT_GPU
            void fill_test(complexDoubleDevice* data);
            void fill_test(complexFloatDevice* data);
            bool check_test(complexDoubleDevice* data);
            bool check_test(complexFloatDevice* data);
            #endif

            void fill_test(complexDoubleHost* data);
            void fill_test(complexFloatHost* data);
            bool check_test(complexDoubleHost* data);
            bool check_test(complexFloatHost* data);

            template<class T>
            bool _test_distribution();
        
        public:
            int ng[3];
            int nlocal;
            int world_size;
            int world_rank;
            int blockSize;
            
            Distribution<MPI_T,REORDER_T>& dist;

            FFTBackend FFTs;

            Dfft(Distribution<MPI_T,REORDER_T>& dist_, bool ks_as_block_);
            ~Dfft();

            bool test_distribution();

            int3 get_ks(int idx);

            #ifdef SWFFT_GPU
            double forward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);
            double forward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
            #endif

            double forward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);
            double forward(complexFloatHost* Buff1, complexFloatHost* Buff2);
            
            #ifdef SWFFT_GPU
            double backward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);
            double backward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
            #endif

            double backward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);
            double backward(complexFloatHost* Buff1, complexFloatHost* Buff2);

    };

}
#ifdef SWFFT_GPU
template<class MPI_T,class FFTBackend>
class AllToAllGPU{
    private:
        A2A::Distribution<MPI_T,A2A::GPUReorder> dist;
        A2A::Dfft<MPI_T,A2A::GPUReorder,FFTBackend> dfft;

    public:
        inline AllToAllGPU(){

        }

        inline AllToAllGPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block=true) : dist(comm,ngx,ngx,ngx,blockSize,ks_as_block), dfft(dist,ks_as_block){

        }

        inline AllToAllGPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block=true) : dist(comm,ngx,ngy,ngz,blockSize,ks_as_block), dfft(dist,ks_as_block){

        }

        inline ~AllToAllGPU(){};

        inline void query(){
            printf("Using AllToAllGPU\n");
            printf("   distribution = [%d %d %d]\n",dist.dims[0],dist.dims[1],dist.dims[2]);
        }

        inline int3 get_ks(int idx){
            return dfft.get_ks(idx);
        }

        inline bool test_distribution(){
            return dfft.test_distribution();
        }

        inline int ngx(){
            return dfft.ng[0];
        }

        inline int ngy(){
            return dfft.ng[1];
        }

        inline int ngz(){
            return dfft.ng[2];
        }

        inline int3 ng(){
            return make_int3(dfft.ng[0],dfft.ng[1],dfft.ng[2]);
        }

        inline int ng(int i){
            return dfft.ng[i];
        }

        inline int buff_sz(){
            return dist.nlocal;
        }

        inline int3 coords(){
            return make_int3(dist.coords[0],dist.coords[1],dist.coords[2]);
        }

        inline int3 dims(){
            return make_int3(dist.dims[0],dist.dims[1],dist.dims[2]);
        }

        inline int rank(){
            return dist.world_rank;
        }

        inline MPI_Comm comm(){
            return dist.comm;
        }

        inline double forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            return dfft.forward(data,scratch);
        }

        inline double forward(complexFloatDevice* data, complexFloatDevice* scratch){
            return dfft.forward(data,scratch);
        }

        inline double forward(complexDoubleHost* data, complexDoubleHost* scratch){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
            return t;
        }

        inline double forward(complexFloatHost* data, complexFloatHost* scratch){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
            return t;
        }

        inline double backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            return dfft.backward(data,scratch);
        }

        inline double backward(complexFloatDevice* data, complexFloatDevice* scratch){
            return dfft.backward(data,scratch);
        }

        inline double backward(complexDoubleHost* data, complexDoubleHost* scratch){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
            return t;
        }

        inline double backward(complexFloatHost* data, complexFloatHost* scratch){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
            return t;
        }

        inline double forward(complexDoubleDevice* data){
            complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * buff_sz());
            double t = forward(data,scratch);
            swfftFree(scratch);
            return t;
        }

        inline double forward(complexFloatDevice* data){
            complexFloatDevice* scratch; swfftAlloc(&scratch,sizeof(complexFloatDevice) * buff_sz());
            double t = forward(data,scratch);
            swfftFree(scratch);
            return t;
        }

        inline double forward(complexDoubleHost* data){
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
            return t;
        }

        inline double forward(complexFloatHost* data){
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
            return t;
        }

        inline double backward(complexDoubleDevice* data){
            complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * buff_sz());
            double t = backward(data,scratch);
            swfftFree(scratch);
            return t;
        }

        inline double backward(complexFloatDevice* data){
            complexFloatDevice* scratch; swfftAlloc(&scratch,sizeof(complexFloatDevice) * buff_sz());
            double t = backward(data,scratch);
            swfftFree(scratch);
            return t;
        }

        inline double backward(complexDoubleHost* data){
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
            return t;
        }

        inline double backward(complexFloatHost* data){
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            double t = backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
            return t;
        }
};
#endif

template<class MPI_T,class FFTBackend>
class AllToAllCPU{
    private:
        A2A::Distribution<MPI_T,A2A::CPUReorder> dist;
        A2A::Dfft<MPI_T,A2A::CPUReorder,FFTBackend> dfft;

    public:
        inline AllToAllCPU(){

        }

        inline AllToAllCPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true) : dist(comm,ngx,ngx,ngx,blockSize,ks_as_block), dfft(dist,ks_as_block){

        }

        inline AllToAllCPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block = true) : dist(comm,ngx,ngy,ngz,blockSize,ks_as_block), dfft(dist,ks_as_block){

        }

        inline ~AllToAllCPU(){};

        inline void query(){
            printf("Using AllToAllCPU\n");
            printf("   distribution = [%d %d %d]\n",dist.dims[0],dist.dims[1],dist.dims[2]);
        }

        inline int3 get_ks(int idx){
            return dfft.get_ks(idx);
        }

        inline bool test_distribution(){
            return dfft.test_distribution();
        }

        inline int ngx(){
            return dfft.ng[0];
        }

        inline int ngy(){
            return dfft.ng[1];
        }

        inline int ngz(){
            return dfft.ng[2];
        }

        inline int3 ng(){
            return make_int3(dfft.ng[0],dfft.ng[1],dfft.ng[2]);
        }

        inline int ng(int i){
            return dfft.ng[i];
        }

        inline int buff_sz(){
            return dist.nlocal;
        }

        inline int3 coords(){
            return make_int3(dist.coords[0],dist.coords[1],dist.coords[2]);
        }

        inline int3 dims(){
            return make_int3(dist.dims[0],dist.dims[1],dist.dims[2]);
        }

        inline int rank(){
            return dist.world_rank;
        }

        inline MPI_Comm comm(){
            return dist.comm;
        }

        inline void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            dfft.forward(data,scratch);
        }

        inline void forward(complexFloatHost* data, complexFloatHost* scratch){
            dfft.forward(data,scratch);
        }

        #ifdef SWFFT_GPU
        inline void forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        inline void forward(complexFloatDevice* data, complexFloatDevice* scratch){
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }
        #endif

        inline void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            dfft.backward(data,scratch);
        }

        inline void backward(complexFloatHost* data, complexFloatHost* scratch){
            dfft.backward(data,scratch);
        }

        #ifdef SWFFT_GPU
        inline void backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        inline void backward(complexFloatDevice* data, complexFloatDevice* scratch){
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }
        #endif

        inline void forward(complexDoubleHost* data){
            complexDoubleHost* scratch; swfftAlloc(&scratch,sizeof(complexDoubleHost) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        inline void forward(complexFloatHost* data){
            complexFloatHost* scratch; swfftAlloc(&scratch,sizeof(complexFloatHost) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        #ifdef SWFFT_GPU
        inline void forward(complexDoubleDevice* data){
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        inline void forward(complexFloatDevice* data){
            complexFloatHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatHost) * buff_sz());
            complexFloatHost* d_data; swfftAlloc(&d_data,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }
        #endif

        inline void backward(complexDoubleHost* data){
            complexDoubleHost* scratch; swfftAlloc(&scratch,sizeof(complexDoubleHost) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        inline void backward(complexFloatHost* data){
            complexFloatHost* scratch; swfftAlloc(&scratch,sizeof(complexFloatHost) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        #ifdef SWFFT_GPU
        inline void backward(complexDoubleDevice* data){
            complexDoubleHost* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleHost* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        inline void backward(complexFloatDevice* data){
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
}
#endif
#endif