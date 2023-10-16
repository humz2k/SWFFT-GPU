#ifdef SWFFT_SMARTMAP
#ifndef SWFFT_SMARTMAP_SEEN
#define SWFFT_SMARTMAP_SEEN

#include "gpu.hpp"
#include "complex-type.h"
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"
#include "mpi_isend_irecv.hpp"
#include "copy_buffers.hpp"
#include "query.hpp"
#include "mpi-map.hpp"

namespace SWFFT{

namespace SMARTMAP{

    class map_1{
        private:
            MPI_Comm comm;
            int3 ng;
            int3 local_grid_size;
            int3 dims;
            int3 coords;
            int world_rank;
            int world_size;
            int nlocal;
            int total_pencils;
            int pencils_per_rank;
            int total_per_rank;
        
        public:
            inline map_1(){}

            inline map_1(MPI_Comm comm_, int3 ng_, int3 local_grid_size_, int3 dims_, int3 coords_, int nlocal_) : comm(comm_), ng(ng_), local_grid_size(local_grid_size_), dims(dims_), coords(coords_), nlocal(nlocal_){
                MPI_Comm_size(comm,&world_size);
                MPI_Comm_rank(comm,&world_rank); 
                total_pencils = ng.x * ng.y;
                pencils_per_rank = total_pencils / world_size;
                total_per_rank = pencils_per_rank * ng.z;

            }

            inline ~map_1(){

            }

            inline map_return_t map(int i){
                
                int this_pencil = i / ng.z;
                int this_pencil_start = world_rank * pencils_per_rank + this_pencil;
                int pencil_x = this_pencil_start / ng.y;
                int pencil_y = this_pencil_start % ng.y;
                int pencil_z = i%ng.z;

                int idx = pencil_x * ng.y * ng.z + pencil_y * ng.z + pencil_z;
                
                int x = idx / (ng.y * ng.z);
                int y = (idx/ng.z)%ng.y;
                int z = idx%ng.z;

                int coord_x = x / local_grid_size.x;
                int coord_y = y / local_grid_size.y;
                int coord_z = z / local_grid_size.z;
                int rank = coord_x * dims.y * dims.z + coord_y * dims.z + coord_z;
                int local_x = x % local_grid_size.x;
                int local_y = y % local_grid_size.y;
                int local_z = z % local_grid_size.z;
                int local_idx = local_x * local_grid_size.y * local_grid_size.z + local_y * local_grid_size.z + local_z;

                //if(!world_rank)
                //    printf("rank %d mapping %d to rank %d idx %d\n",world_rank,i,rank,local_idx);

                return make_map_return(rank,local_idx,i);
            }

    };

    class map_2{
        private:
            MPI_Comm comm;
            int3 ng;
            int3 local_grid_size;
            int3 dims;
            int3 coords;
            int world_rank;
            int world_size;
            int nlocal;
            int total_pencils;
            int pencils_per_rank;
            int total_per_rank;
        
        public:
            inline map_2(){};

            inline map_2(MPI_Comm comm_, int3 ng_, int3 local_grid_size_, int3 dims_, int3 coords_, int nlocal_) : comm(comm_), ng(ng_), local_grid_size(local_grid_size_), dims(dims_), coords(coords_), nlocal(nlocal_){
                MPI_Comm_size(comm,&world_size);
                MPI_Comm_rank(comm,&world_rank);
                total_pencils = ng.x * ng.z;
                pencils_per_rank = total_pencils / world_size;
                total_per_rank = pencils_per_rank * ng.y;

            }

            inline ~map_2(){

            }

            inline map_return_t map(int i){
                
                int this_pencil = i / ng.y;
                int this_pencil_start = world_rank * pencils_per_rank + this_pencil;
                int pencil_x = this_pencil_start / ng.z;
                int pencil_z = this_pencil_start % ng.z;
                int pencil_y = i%ng.y;

                int idx = pencil_x * ng.y * ng.z + pencil_y * ng.z + pencil_z;

                int pencil_id = pencil_x * ng.y + pencil_y;
                int rank = pencil_id / ((ng.x*ng.y)/world_size);
                int local_idx = idx % (((ng.x*ng.y)/world_size) * ng.z);

                return make_map_return(rank,local_idx,i);
                
            }

    };

    class map_3{
        private:
            MPI_Comm comm;
            int3 ng;
            int3 local_grid_size;
            int3 dims;
            int3 coords;
            int world_rank;
            int world_size;
            int nlocal;
            int total_pencils;
            int pencils_per_rank;
            int total_per_rank;
        
        public:
            inline map_3(){}

            inline map_3(MPI_Comm comm_, int3 ng_, int3 local_grid_size_, int3 dims_, int3 coords_, int nlocal_) : comm(comm_), ng(ng_), local_grid_size(local_grid_size_), dims(dims_), coords(coords_), nlocal(nlocal_){
                MPI_Comm_size(comm,&world_size);
                MPI_Comm_rank(comm,&world_rank);
                total_pencils = ng.z * ng.y;
                pencils_per_rank = total_pencils / world_size;
                total_per_rank = pencils_per_rank * ng.x;

            }

            inline ~map_3(){

            }

            inline map_return_t map(int i){
                
                int this_pencil = i / ng.x;
                int this_pencil_start = world_rank * pencils_per_rank + this_pencil;
                int pencil_z = this_pencil_start / ng.y;
                int pencil_y = this_pencil_start % ng.y;
                int pencil_x = i%ng.x;

                int idx = pencil_x * ng.y * ng.z + pencil_y * ng.z + pencil_z;

                int pencil_id = pencil_x * ng.z + pencil_z;
                int rank = pencil_id / ((ng.x*ng.z)/world_size);
                int local_idx = idx % (((ng.x*ng.z)/world_size) * ng.y);

                return make_map_return(rank,local_idx,i);
                
            }

    };

    template<class MPI_T, class FFTBackend>
    class Dfft{
        private:
            map_1 map1;
            map_2 map2;
            map_3 map3;
            SmartMap<map_1>* sm1;
            SmartMap<map_2>* sm2;
            SmartMap<map_3>* sm3;
            FFTBackend ffts;

            template<class T>
            void _forward(T* data, T* scratch);

            template<class T>
            void _backward(T* data, T* scratch);

        public:
            MPI_Comm comm;
            int ng[3];
            int local_grid_size[3];
            int comm_rank;
            int comm_size;
            int dims[3];
            int coords[3];
            int nlocal;
            int local_coords_start[3];

            Dfft(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize_);

            ~Dfft();

            void forward(complexDoubleHost* data, complexDoubleHost* scratch);
            void forward(complexFloatHost* data, complexFloatHost* scratch);

            void backward(complexDoubleHost* data, complexDoubleHost* scratch);
            void backward(complexFloatHost* data, complexFloatHost* scratch);

            int buff_sz();

    };

}

template<class MPI_T, class FFTBackend>
class SmartFFT{
    private:
        SMARTMAP::Dfft<MPI_T,FFTBackend> dfft;
    
    public:
        inline SmartFFT(){

        }

        inline SmartFFT(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true) : dfft(comm,ngx,ngx,ngx,blockSize){

        }

        inline SmartFFT(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block = true) : dfft(comm,ngx,ngy,ngz,blockSize){

        }

        inline ~SmartFFT(){};

        inline void query(){
            printf("Using SmartFFT\n");
        }

        inline void set_nsends(int x){
            
        }

        inline void set_delegate(int r){
            
        }

        inline void synchronize(){
            
        }

        inline int3 local_ng(){
            return make_int3(dfft.local_grid_size[0],dfft.local_grid_size[1],dfft.local_grid_size[2]);
        }

        inline int local_ng(int i){
            return dfft.local_grid_size[i];
        }

        inline int3 get_ks(int idx){
            return make_int3(0,0,0);
        }

        inline int3 get_rs(int idx){
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
            return dfft.buff_sz();
        }

        inline int3 coords(){
            return make_int3(dfft.coords[0],dfft.coords[1],dfft.coords[2]);
        }

        inline int3 dims(){
            return make_int3(dfft.dims[0],dfft.dims[1],dfft.dims[2]);
        }

        inline int rank(){
            return dfft.comm_rank;
        }

        inline MPI_Comm comm(){
            return dfft.comm;
        }

        inline void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            dfft.forward(data,scratch);
        }

        inline void forward(complexFloatHost* data, complexFloatHost* scratch){
            dfft.forward(data,scratch);
        }

        inline void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            dfft.backward(data,scratch);
        }

        inline void backward(complexFloatHost* data, complexFloatHost* scratch){
            dfft.backward(data,scratch);
        }

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
        inline void forward(complexDoubleDevice* data){
            complexDoubleHost* h_data; swfftAlloc(&h_data,sizeof(complexDoubleHost) * buff_sz());
            gpuMemcpy(h_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            forward(h_data);
            gpuMemcpy(data,h_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(h_data);
        }

        inline void forward(complexFloatDevice* data){
            complexFloatHost* h_data; swfftAlloc(&h_data,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(h_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            forward(h_data);
            gpuMemcpy(data,h_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(h_data);
        }

        inline void backward(complexDoubleDevice* data){
            complexDoubleHost* h_data; swfftAlloc(&h_data,sizeof(complexDoubleHost) * buff_sz());
            gpuMemcpy(h_data,data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            backward(h_data);
            gpuMemcpy(data,h_data,sizeof(complexDoubleDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(h_data);
        }

        inline void backward(complexFloatDevice* data){
            complexFloatHost* h_data; swfftAlloc(&h_data,sizeof(complexFloatHost) * buff_sz());
            gpuMemcpy(h_data,data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyDeviceToHost);
            backward(h_data);
            gpuMemcpy(data,h_data,sizeof(complexFloatDevice) * buff_sz(),gpuMemcpyHostToDevice);
            swfftFree(h_data);
        }
        #endif
        
        #ifdef SWFFT_GPU
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
        #endif


};

template<> 
inline const char* queryName<SmartFFT>(){
    return "SmartFFT";
}

}


#endif
#endif