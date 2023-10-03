#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
#ifdef SWFFT_GPUDELEGATE
#ifndef SWFFT_GPUDELEGATE_SEEN
#define SWFFT_GPUDELEGATE_SEEN

#include "gpu.hpp"
#include "complex-type.h"
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "mpi_isend_irecv.hpp"
#include "query.hpp"

#include "gpudelegate_reorder.hpp"

namespace SWFFT{

namespace GPUDELEGATE{

    template<class T>
    class copyBuffers{
        private:
            T* dest;
            T* src;
            int n;
            #ifdef SWFFT_GPU
            gpuEvent_t event;
            #endif
            
        public:
            inline copyBuffers(T* dest_, T* src_, int n_);
            inline ~copyBuffers();
            inline void wait();
    };

    template<class T>
    inline copyBuffers<T>::copyBuffers(T* dest_, T* src_, int n_) : dest(dest_), src(src_), n(n_){
        for (int i = 0; i < n; i++){
            dest[i] = src[i];
        }
    }

    template<class T>
    inline copyBuffers<T>::~copyBuffers(){

    }

    template<class T>
    inline void copyBuffers<T>::wait(){

    }
    #ifdef SWFFT_GPU

    template<>
    inline copyBuffers<complexDoubleDevice>::~copyBuffers(){

    }

    template<>
    inline copyBuffers<complexFloatDevice>::~copyBuffers(){

    }

    template<>
    inline copyBuffers<complexDoubleDevice>::copyBuffers(complexDoubleDevice* dest_, complexDoubleDevice* src_, int n_) : dest(dest_), src(src_), n(n_){
        gpuEventCreate(&event);
        gpuMemcpyAsync(dest,src,n * sizeof(complexDoubleDevice),gpuMemcpyDeviceToDevice);
        gpuEventRecord(event);
    }

    template<>
    inline void copyBuffers<complexDoubleDevice>::wait(){
        gpuEventSynchronize(event);
        gpuEventDestroy(event);
    }

    template<>
    inline copyBuffers<complexFloatDevice>::copyBuffers(complexFloatDevice* dest_, complexFloatDevice* src_, int n_) : dest(dest_), src(src_), n(n_){
        gpuEventCreate(&event);
        gpuMemcpyAsync(dest,src,n * sizeof(complexFloatDevice),gpuMemcpyDeviceToDevice);
        gpuEventRecord(event);
    }

    template<>
    inline void copyBuffers<complexFloatDevice>::wait(){
        gpuEventSynchronize(event);
        gpuEventDestroy(event);
    }
    #endif


    /*template<class MPI_T, class T>
    class Isend{

    };

    template<class MPI_T, class T>
    class Irecv{

    };

    #ifdef SWFFT_GPU
    template<>
    class Isend<CPUMPI,complexDoubleDevice>{
        private:
            CPUIsend<complexDoubleDevice>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexDoubleDevice>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Isend<CPUMPI,complexFloatDevice>{
        private:
            CPUIsend<complexFloatDevice>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexFloatDevice>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexDoubleDevice>{
        private:
            CPUIrecv<complexDoubleDevice>* raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(CPUIrecv<complexDoubleDevice>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexFloatDevice>{
        private:
            CPUIrecv<complexFloatDevice>* raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(CPUIrecv<complexFloatDevice>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };
    #endif

    template<>
    class Isend<CPUMPI,complexFloatHost>{
        private:
            CPUIsend<complexFloatHost>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexFloatHost>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Isend<CPUMPI,complexDoubleHost>{
        private:
            CPUIsend<complexDoubleHost>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexDoubleHost>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexFloatHost>{
        private:
            CPUIrecv<complexFloatHost>* raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(CPUIrecv<complexFloatHost>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexDoubleHost>{
        private:
            CPUIrecv<complexDoubleHost>* raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(CPUIrecv<complexDoubleHost>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;}
    };*/

    template<class MPI_T, class FFTBackend>
    class Dfft{
        private:
            template<class T>
            void _fft(T* buff1, int direction);

            template<class T>
            void _fftNoReorder(T* buff1, int direction);

            void execFFT(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int direction);
            void execFFT(complexFloatDevice* buff1, complexFloatDevice* buff2, int direction);

            void* _sends;
            void* _recvs;
            void* _fftBuff1;
            void* _fftBuff2;
            int last_t;
            int set;

            void set_last_t(complexDoubleDevice* a);
            void set_last_t(complexFloatDevice* a);

        public:
            MPI_T mpi;
            MPI_Comm comm;
            int ng[3];
            int3 int3_ng;
            int local_grid_size[3];
            int3 int3_local_grid_size;
            int dims[3];
            int3 int3_dims;
            int coords[3];
            int3 int3_coords;
            int local_coords_start[3];
            int blockSize;
            int nlocal;
            int world_rank;
            int world_size;
            int delegate;
            int nsends;
            gpufftHandle planDouble;
            gpufftHandle planFloat;

            Dfft(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_, int nsends_ = 1, int delegate_ = 0);

            ~Dfft();

            void fft(complexDoubleDevice* buff1, int direction);
            void fft(complexFloatDevice* buff1, int direction);

            void forward(complexDoubleDevice* buff1);
            void backward(complexDoubleDevice* buff1);
            void forward(complexFloatDevice* buff1);
            void backward(complexFloatDevice* buff1);
            void synchronize();

            int buff_sz();

    };

}

template<class MPI_T, class FFTBackend>
class GPUDelegate{
    private:
        GPUDELEGATE::Dfft<MPI_T,FFTBackend> dfft;
    
    public:
        inline GPUDelegate(){

        }

        inline GPUDelegate(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true) : dfft(comm,ngx,ngx,ngx,blockSize){

        }

        inline GPUDelegate(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block = true) : dfft(comm,ngx,ngy,ngz,blockSize){

        }

        inline ~GPUDelegate(){};

        inline void query(){
            printf("Using GPUDelegate\n");
        }

        inline void set_nsends(int x){
            dfft.nsends = x;
        }

        inline void set_delegate(int r){
            dfft.delegate = r;
        }

        inline void synchronize(){
            dfft.synchronize();
        }

        inline int3 local_ng(){
            return make_int3(dfft.local_grid_size[0],dfft.local_grid_size[1],dfft.local_grid_size[2]);
        }

        inline int local_ng(int i){
            return dfft.local_grid_size[i];
        }

        inline int3 get_ks(int idx){
            int3 start;
            start.x = dfft.local_coords_start[0];
            start.y = dfft.local_coords_start[1];
            start.z = dfft.local_coords_start[2];
            int3 this_idx;
            this_idx.x = idx / (dfft.local_grid_size[1] * dfft.local_grid_size[2]);
            this_idx.y = (idx - (this_idx.x * dfft.local_grid_size[1] * dfft.local_grid_size[2])) / dfft.local_grid_size[2];
            this_idx.z = (idx - (this_idx.x * dfft.local_grid_size[1] * dfft.local_grid_size[2])) - this_idx.y * dfft.local_grid_size[2];
            int3 out;
            out.x = start.x + this_idx.x;
            out.y = start.y + this_idx.y;
            out.z = start.z + this_idx.z;
            return out;
        }

        inline int3 get_rs(int idx){
            int3 start;
            start.x = dfft.local_coords_start[0];
            start.y = dfft.local_coords_start[1];
            start.z = dfft.local_coords_start[2];
            int3 this_idx;
            this_idx.x = idx / (dfft.local_grid_size[1] * dfft.local_grid_size[2]);
            this_idx.y = (idx - (this_idx.x * dfft.local_grid_size[1] * dfft.local_grid_size[2])) / dfft.local_grid_size[2];
            this_idx.z = (idx - (this_idx.x * dfft.local_grid_size[1] * dfft.local_grid_size[2])) - this_idx.y * dfft.local_grid_size[2];
            int3 out;
            out.x = start.x + this_idx.x;
            out.y = start.y + this_idx.y;
            out.z = start.z + this_idx.z;
            return out;
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
            return dfft.buff_sz();
        }

        inline int3 coords(){
            return make_int3(dfft.coords[0],dfft.coords[1],dfft.coords[2]);
        }

        inline int3 dims(){
            return make_int3(dfft.dims[0],dfft.dims[1],dfft.dims[2]);
        }

        inline int rank(){
            return dfft.world_rank;
        }

        inline MPI_Comm comm(){
            return dfft.comm;
        }

        inline void forward(complexDoubleDevice* data){
            dfft.forward(data);
        }

        inline void forward(complexFloatDevice* data){
            dfft.forward(data);
        }

        inline void forward(complexDoubleHost* data){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleHost) * dfft.nlocal);
            gpuMemcpy(d_data,data,sizeof(complexDoubleHost) * dfft.nlocal,gpuMemcpyHostToDevice);
            forward(d_data);
            synchronize();
            gpuMemcpy(data,d_data,sizeof(complexDoubleHost) * dfft.nlocal,gpuMemcpyDeviceToHost);
            swfftFree(d_data);
        }

        inline void forward(complexFloatHost* data){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * dfft.nlocal);
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * dfft.nlocal,gpuMemcpyHostToDevice);
            forward(d_data);
            synchronize();
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * dfft.nlocal,gpuMemcpyDeviceToHost);
            swfftFree(d_data);
        }

        inline void backward(complexDoubleDevice* data){
            dfft.backward(data);
        }

        inline void backward(complexFloatDevice* data){
            dfft.backward(data);
        }

        inline void backward(complexDoubleHost* data){
            complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleHost) * dfft.nlocal);
            gpuMemcpy(d_data,data,sizeof(complexDoubleHost) * dfft.nlocal,gpuMemcpyHostToDevice);
            backward(d_data);
            synchronize();
            gpuMemcpy(data,d_data,sizeof(complexDoubleHost) * dfft.nlocal,gpuMemcpyDeviceToHost);
            swfftFree(d_data);
        }

        inline void backward(complexFloatHost* data){
            complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * dfft.nlocal);
            gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * dfft.nlocal,gpuMemcpyHostToDevice);
            backward(d_data);
            synchronize();
            gpuMemcpy(data,d_data,sizeof(complexFloatDevice) * dfft.nlocal,gpuMemcpyDeviceToHost);
            swfftFree(d_data);
        }

        inline void forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            forward(data);
        }

        inline void forward(complexFloatDevice* data, complexFloatDevice* scratch){
            forward(data);
        }

        inline void backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            backward(data);
        }

        inline void backward(complexFloatDevice* data, complexFloatDevice* scratch){
            backward(data);
        }

        inline void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            forward(data);
        }

        inline void forward(complexFloatHost* data, complexFloatHost* scratch){
            forward(data);
        }

        inline void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            backward(data);
        }

        inline void backward(complexFloatHost* data, complexFloatHost* scratch){
            backward(data);
        }


};

template<> 
inline const char* queryName<GPUDelegate>(){
    return "GPUDelegate";
}

}


#endif
#endif
#endif
#endif