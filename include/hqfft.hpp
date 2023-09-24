#ifdef SWFFT_HQFFT
#ifndef SWFFT_HQFFT_SEEN
#define SWFFT_HQFFT_SEEN

#include <mpi.h>
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "hqfft_reorder.hpp"

namespace SWFFT{

namespace HQFFT{

    template<class MPI_T, class T>
    class Isend{

    };

    template<class MPI_T, class T>
    class Irecv{

    };

    #ifdef SWFFT_GPU
    template<>
    class Isend<CPUMPI,complexDoubleDevice>{
        private:
            CPUIsend<complexDoubleDevice> raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexDoubleDevice> in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};
    };

    template<>
    class Isend<CPUMPI,complexFloatDevice>{
        private:
            CPUIsend<complexFloatDevice> raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexFloatDevice> in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};
    };

    template<>
    class Irecv<CPUMPI,complexDoubleDevice>{
        private:
            CPUIrecv<complexDoubleDevice> raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(CPUIrecv<complexDoubleDevice> in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};

            inline void finalize(){raw.finalize();};
    };

    template<>
    class Irecv<CPUMPI,complexFloatDevice>{
        private:
            CPUIrecv<complexFloatDevice> raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(CPUIrecv<complexFloatDevice> in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};

            inline void finalize(){raw.finalize();};
    };
    #endif

    template<>
    class Isend<CPUMPI,complexFloatHost>{
        private:
            CPUIsend<complexFloatHost> raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexFloatHost> in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};
    };

    template<>
    class Isend<CPUMPI,complexDoubleHost>{
        private:
            CPUIsend<complexDoubleHost> raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexDoubleHost> in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};
    };

    template<>
    class Irecv<CPUMPI,complexFloatHost>{
        private:
            CPUIrecv<complexFloatHost> raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(CPUIrecv<complexFloatHost> in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};

            inline void finalize(){raw.finalize();};
    };

    template<>
    class Irecv<CPUMPI,complexDoubleHost>{
        private:
            CPUIrecv<complexDoubleHost> raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(CPUIrecv<complexDoubleHost> in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw.execute();};

            inline void wait(){raw.wait();};

            inline void finalize(){raw.finalize();}
    };

    template<class MPI_T>
    class CollectiveCommunicator{
        public:
            MPI_T mpi;

            CollectiveCommunicator();
            ~CollectiveCommunicator();

            //template<class T>
            //void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            void query();

    };
    
    template<class MPI_T>
    class AllToAll : public CollectiveCommunicator<MPI_T>{
        private:
            template<class T>
            inline void _alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        public:

            #ifdef SWFFT_GPU
            void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm);
            #endif

            void alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm);

            void query();
    };

    template<class MPI_T>
    class PairSends : public CollectiveCommunicator<MPI_T>{
        private:
            template<class T>
            inline void _alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

        public:
            #ifdef SWFFT_GPU
            void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv, MPI_Comm comm);
            #endif

            void alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv, MPI_Comm comm);
            void alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv, MPI_Comm comm);

            void query();
    };

    template<template<class> class Communicator, class MPI_T, class REORDER_T>
    class Distribution{
        private:
            template<class T>
            void _pencils_1(T* buff1, T* buff2);

            template<class T>
            void _pencils_2(T* buff1, T* buff2);

            template<class T>
            void _pencils_3(T* buff1, T* buff2);

            template<class T>
            void _return_pencils(T* buff1, T* buff2);

        public:
            int ng[3];
            int nlocal;
            int world_size;
            int world_rank;
            int local_grid_size[3];
            int dims[3];
            int coords[3];
            int local_coords_start[3];
            MPI_Comm world_comm;
            MPI_Comm distcomms[4];

            Communicator<MPI_T> CollectiveComm;
            REORDER_T reorder;

            int blockSize;

            Distribution(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);
            ~Distribution();

            void pencils_1(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void pencils_1(complexFloatHost* buff1, complexFloatHost* buff2);
            void pencils_2(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void pencils_2(complexFloatHost* buff1, complexFloatHost* buff2);
            void pencils_3(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void pencils_3(complexFloatHost* buff1, complexFloatHost* buff2);
            void return_pencils(complexDoubleHost* buff1, complexDoubleHost* buff2);
            void return_pencils(complexFloatHost* buff1, complexFloatHost* buff2);

            #ifdef SWFFT_GPU
            void pencils_1(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void pencils_1(complexFloatDevice* buff1, complexFloatDevice* buff2);
            void pencils_2(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void pencils_2(complexFloatDevice* buff1, complexFloatDevice* buff2);
            void pencils_3(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void pencils_3(complexFloatDevice* buff1, complexFloatDevice* buff2);
            void return_pencils(complexDoubleDevice* buff1, complexDoubleDevice* buff2);
            void return_pencils(complexFloatDevice* buff1, complexFloatDevice* buff2);
            #endif

            template<class T>
            void reshape_1(T* buff1, T* buff2);

            template<class T>
            void unreshape_1(T* buff1, T* buff2);

            template<class T>
            void reshape_2(T* buff1, T* buff2);

            template<class T>
            void unreshape_2(T* buff1, T* buff2);

            template<class T>
            void reshape_3(T* buff1, T* buff2);

            template<class T>
            void unreshape_3(T* buff1, T* buff2);

            template<class T>
            void reshape_final(T* buff1, T* buff2, int ny, int nz);

            //template<class T>
            //void alltoall(T* src, T* dest, int n_recv, MPI_Comm comm);

            int buff_sz();
    };

    template<template<template<class> class,class,class> class Dist, class REORDER_T, template<class> class CollectiveComm, class MPI_T, class FFTBackend>
    class Dfft{
        private:
            template<class T>
            inline void _forward(T* buff1, T* buff2);

            template<class T>
            inline void _backward(T* buff1, T* buff2);

        public:
            Dist<CollectiveComm,MPI_T,REORDER_T>& dist;
            FFTBackend FFTs;
            int ng[3];
            int nlocal;

            Dfft(Dist<CollectiveComm,MPI_T,REORDER_T>& dist_);
            ~Dfft();
            
            #ifdef SWFFT_GPU
            void forward(complexDoubleDevice* data, complexDoubleDevice* scratch);
            void forward(complexFloatDevice* data, complexFloatDevice* scratch);
            void backward(complexDoubleDevice* data, complexDoubleDevice* scratch);
            void backward(complexFloatDevice* data, complexFloatDevice* scratch);
            #endif

            void forward(complexDoubleHost* data, complexDoubleHost* scratch);
            void forward(complexFloatHost* data, complexFloatHost* scratch);
            void backward(complexDoubleHost* data, complexDoubleHost* scratch);
            void backward(complexFloatHost* data, complexFloatHost* scratch);

            int buff_sz();
    };

}
#ifdef SWFFT_GPU
template<class MPI_T, class FFTBackend>
class HQA2AGPU{
    private:
        HQFFT::Distribution<HQFFT::AllToAll,MPI_T,HQFFT::GPUReshape> dist;
        HQFFT::Dfft<HQFFT::Distribution,HQFFT::GPUReshape,HQFFT::AllToAll,MPI_T,FFTBackend> dfft;
    
    public:
        inline HQA2AGPU(){

        }

        inline HQA2AGPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true) : dist(comm,ngx,ngx,ngx,blockSize), dfft(dist){

        }

        inline HQA2AGPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block = true) : dist(comm,ngx,ngy,ngz,blockSize), dfft(dist){

        }

        inline ~HQA2AGPU(){};

        inline void query(){
            printf("Using AllToAllCPU\n");
            printf("   distribution = [%d %d %d]\n",dist.dims[0],dist.dims[1],dist.dims[2]);
        }

        inline int3 get_ks(int idx){
            return make_int3(0,0,0);
        }

        inline bool test_distribution(){
            return false;
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
            return dist.world_comm;
        }

        inline void forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            dfft.forward(data,scratch);
        }

        inline void forward(complexFloatDevice* data, complexFloatDevice* scratch){
            dfft.forward(data,scratch);
        }

        inline void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        inline void forward(complexFloatHost* data, complexFloatHost* scratch){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        inline void backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            dfft.backward(data,scratch);
        }

        inline void backward(complexFloatDevice* data, complexFloatDevice* scratch){
            dfft.backward(data,scratch);
        }

        inline void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        inline void backward(complexFloatHost* data, complexFloatHost* scratch){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            dfft.backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_data);
            swfftFree(d_scratch);
        }

        inline void forward(complexDoubleDevice* data){
            complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        inline void forward(complexFloatDevice* data){
            complexFloatDevice* scratch; swfftAlloc(&scratch,sizeof(complexFloatDevice) * buff_sz());
            forward(data,scratch);
            swfftFree(scratch);
        }

        inline void forward(complexDoubleHost* data){
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        inline void forward(complexFloatHost* data){
            complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * buff_sz());
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            forward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        inline void backward(complexDoubleDevice* data){
            complexDoubleDevice* scratch; swfftAlloc(&scratch,sizeof(complexDoubleDevice) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        inline void backward(complexFloatDevice* data){
            complexFloatDevice* scratch; swfftAlloc(&scratch,sizeof(complexFloatDevice) * buff_sz());
            backward(data,scratch);
            swfftFree(scratch);
        }

        inline void backward(complexDoubleHost* data){
            complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * buff_sz());
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * buff_sz());
            gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            backward(d_data,d_scratch);
            gpuMemcpy(data,d_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            swfftFree(d_scratch);
            swfftFree(d_data);
        }

        inline void backward(complexFloatHost* data){
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

}

#endif
#endif