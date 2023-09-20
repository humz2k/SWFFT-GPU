#ifdef ALLTOALL
#include "alltoall.hpp"

namespace A2A{
    template<class MPI_T,class REORDER_T,class FFTBackend>
    Dfft<MPI_T,REORDER_T,FFTBackend>::Dfft(Distribution<MPI_T,REORDER_T>& dist_, bool ks_as_block_) : dist(dist_), ks_as_block(ks_as_block_){
        ng[0] = dist.ng[0];
        ng[1] = dist.ng[1];
        ng[2] = dist.ng[2];

        nlocal = dist.nlocal;

        world_size = dist.world_size;
        world_rank = dist.world_rank;

        blockSize = dist.blockSize;
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    Dfft<MPI_T,REORDER_T,FFTBackend>::~Dfft(){}

    template<class MPI_T,class REORDER_T,class FFTBackend>
    int3 Dfft<MPI_T,REORDER_T,FFTBackend>::get_ks(int idx){
        if (ks_as_block){
            int3 local_idx;
            local_idx.x = idx / (dist.local_grid_size[1] * dist.local_grid_size[2]);
            local_idx.y = (idx - local_idx.x * (dist.local_grid_size[1] * dist.local_grid_size[2])) / dist.local_grid_size[2];
            local_idx.z = (idx - local_idx.x * (dist.local_grid_size[1] * dist.local_grid_size[2])) - (local_idx.y * dist.local_grid_size[2]);
            int3 global_idx = make_int3(dist.local_coordinates_start[0] + local_idx.x, dist.local_coordinates_start[1] + local_idx.y, dist.local_coordinates_start[2] + local_idx.z);
            return global_idx;
        }
        int mini_pencil_size = dist.local_grid_size[1];
        int global_mini_pencil_offset = world_size * mini_pencil_size;
        int mini_pencils_per_rank = (nlocal / world_size) / mini_pencil_size;

        int sub_mini_pencil_idx = idx % mini_pencil_size;
        int my_pencil_start = idx - sub_mini_pencil_idx;
        int rank = (my_pencil_start / mini_pencil_size) % world_size;

        int my_mini_pencil_offset = my_pencil_start - rank * mini_pencil_size;
        int local_mini_pencil_id = my_mini_pencil_offset / global_mini_pencil_offset;

        int global_mini_pencil_id = rank * mini_pencils_per_rank + local_mini_pencil_id;

        int i = global_mini_pencil_id * mini_pencil_size + sub_mini_pencil_idx;

        int3 rank_coords;
        rank_coords.y = rank / (dist.dims[0] * dist.dims[2]);
        rank_coords.z = (rank - rank_coords.y * (dist.dims[0] * dist.dims[2])) / dist.dims[0];
        rank_coords.x = (rank - rank_coords.y * (dist.dims[0] * dist.dims[2])) - rank_coords.z * dist.dims[0];

        int global_start = rank_coords.x * dist.local_grid_size[1] * dist.local_grid_size[2] + rank_coords.y * dist.local_grid_size[2] + rank_coords.z;
        int global_1d_idx = global_start + world_rank * (nlocal / world_size) + (i % (nlocal / world_size));

        int3 local_idx;
        local_idx.x = i;
        local_idx.z = 0;//(global_1d_idx - local_idx.x * (ng[1] * ng[2])) / ng[1];
        local_idx.y = 0;//(global_1d_idx - local_idx.x * (ng[1] * ng[2])) - (local_idx.z * ng[1]);
        //int3 global_idx = make_int3(dist.local_coordinates_start[0] + local_idx.x, dist.local_coordinates_start[1] + local_idx.y, dist.local_coordinates_start[2] + local_idx.z);
        return local_idx;

    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::fill_test(complexDoubleHost* data){
        int my_start = (dist.world_rank * nlocal) * 2;
        for (int i = 0; i < nlocal; i++){
            data[i].x = my_start + i*2;
            data[i].y = my_start + i*2 + 1;
        }
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::fill_test(complexFloatHost* data){
        int my_start = (dist.world_rank * nlocal) * 2;
        for (int i = 0; i < nlocal; i++){
            data[i].x = my_start + i*2;
            data[i].y = my_start + i*2 + 1;
        }
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    bool Dfft<MPI_T,REORDER_T,FFTBackend>::check_test(complexDoubleHost* data){
        int my_start = (dist.world_rank * nlocal) * 2;
        for (int i = 0; i < nlocal; i++){
            if (data[i].x != (my_start + i*2))return false;
            if (data[i].y != (my_start + i*2 + 1))return false;
        }
        return true;
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    bool Dfft<MPI_T,REORDER_T,FFTBackend>::check_test(complexFloatHost* data){
        int my_start = (dist.world_rank * nlocal) * 2;
        for (int i = 0; i < nlocal; i++){
            if (data[i].x != (my_start + i*2))return false;
            if (data[i].y != (my_start + i*2 + 1))return false;
        }
        return true;
    }

    #ifdef GPU
    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::fill_test(complexDoubleDevice* data){
        complexDoubleHost* h_data; swfftAlloc(&h_data, sizeof(complexDoubleHost) * nlocal);
        fill_test(h_data);
        gpuMemcpy(data,h_data,sizeof(complexDoubleDevice) * nlocal,gpuMemcpyHostToDevice);
        swfftFree(h_data);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::fill_test(complexFloatDevice* data){
        complexFloatHost* h_data; swfftAlloc(&h_data, sizeof(complexFloatDevice) * nlocal);
        fill_test(h_data);
        gpuMemcpy(data,h_data,sizeof(complexFloatDevice) * nlocal,gpuMemcpyHostToDevice);
        swfftFree(h_data);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    bool Dfft<MPI_T,REORDER_T,FFTBackend>::check_test(complexDoubleDevice* data){
        complexDoubleHost* h_data; swfftAlloc(&h_data, sizeof(complexDoubleHost) * nlocal);
        gpuMemcpy(h_data,data,sizeof(complexDoubleDevice) * nlocal,gpuMemcpyDeviceToHost);
        bool out = check_test(h_data);
        swfftFree(h_data);
        return out;
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    bool Dfft<MPI_T,REORDER_T,FFTBackend>::check_test(complexFloatDevice* data){
        complexFloatHost* h_data; swfftAlloc(&h_data, sizeof(complexFloatHost) * nlocal);
        gpuMemcpy(h_data,data,sizeof(complexFloatHost) * nlocal,gpuMemcpyDeviceToHost);
        bool out = check_test(h_data);
        swfftFree(h_data);
        return out;
    }
    #endif

    template<class MPI_T,class REORDER_T,class FFTBackend>
    template<class T>
    bool Dfft<MPI_T,REORDER_T,FFTBackend>::_test_distribution(){

        T* data; swfftAlloc(&data,sizeof(T) * nlocal);
        T* scratch; swfftAlloc(&scratch,sizeof(T) * nlocal);

        fill_test(data);
        bool out = false;
        if (ks_as_block){
            #pragma GCC unroll 3
            for (int i = 0; i < 3; i++){
                dist.getPencils(data,scratch,i);
                dist.reorder(data,scratch,i,0);

                //gpuDeviceSynchronize();
                dist.copy(scratch,data);

                dist.reorder(data,scratch,i,1);
                //gpuDeviceSynchronize();
                dist.returnPencils(data,scratch,i);

                dist.shuffle_indices(data,scratch,i);
                //gpuDeviceSynchronize();

            }
            out = check_test(data);
        } else {

            for (int i = 0; i < 2; i++){
                dist.getPencils(data,scratch,i);
                dist.reorder(data,scratch,i,0);
                
                dist.copy(scratch,data);

                dist.reorder(data,scratch,i,1);
                dist.returnPencils(data,scratch,i);
                dist.shuffle_indices(data,scratch,i);
            }
            dist.getPencils(data,scratch,2);
            dist.reorder(data,scratch,2,0);

            dist.copy(scratch,data);

            dist.copy(data,scratch);

            dist.copy(scratch,data);

            dist.reorder(data,scratch,2,1);
            dist.returnPencils(data,scratch,2);
            dist.shuffle_indices(data,scratch,2);

            dist.getPencils(data,scratch,0);
            dist.reorder(data,scratch,0,0);

            dist.copy(scratch,data);

            dist.reorder(data,scratch,0,1);
            dist.returnPencils(data,scratch,0);
            dist.shuffle_indices(data,scratch,0);

            dist.getPencils(data,scratch,1);
            dist.reorder(data,scratch,1,0);

            dist.copy(scratch,data);

            dist.reorder(data,scratch,1,1);
            dist.returnPencils(data,scratch,1);
            dist.shuffle_indices(data,scratch,3);

            out = check_test(data);
            
        }

        swfftFree(data);
        swfftFree(scratch);

        return out;

    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    bool Dfft<MPI_T,REORDER_T,FFTBackend>::test_distribution(){
        bool out = _test_distribution<complexDoubleHost>();
        out = out && _test_distribution<complexFloatHost>();
        #ifdef GPU
        out = out && _test_distribution<complexDoubleDevice>();
        out = out && _test_distribution<complexFloatDevice>();
        #endif
        return out;
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    template<class T>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::fft(T* data, T* scratch, fftdirection direction){
        double start = MPI_Wtime();
        if (ks_as_block){
            #pragma GCC unroll 3
            for (int i = 0; i < 3; i++){
                dist.getPencils(data,scratch,i);
                dist.reorder(data,scratch,i,0);

                //gpuDeviceSynchronize();
                int nFFTs = (nlocal / ng[i]);
                if (direction == FFT_FORWARD){
                    FFTs.forward(data,scratch,ng[i],nFFTs);
                } else {
                    FFTs.backward(data,scratch,ng[i],nFFTs);
                }

                dist.reorder(data,scratch,i,1);
                //gpuDeviceSynchronize();
                dist.returnPencils(data,scratch,i);

                dist.shuffle_indices(data,scratch,i);
                //gpuDeviceSynchronize();

            }
        } else {

            if (direction == FFT_FORWARD){
                for (int i = 0; i < 2; i++){
                    dist.getPencils(data,scratch,i);
                    dist.reorder(data,scratch,i,0);
                    
                    int nFFTs = (nlocal / ng[i]);
                    FFTs.forward(data,scratch,ng[i],nFFTs);

                    dist.reorder(data,scratch,i,1);
                    dist.returnPencils(data,scratch,i);
                    dist.shuffle_indices(data,scratch,i);
                }
                dist.getPencils(data,scratch,2);
                dist.reorder(data,scratch,2,0);
                int nFFTs = (nlocal / ng[2]);
                FFTs.forward(data,scratch,ng[2],nFFTs);
                dist.copy(data,scratch);

            } else {
                int nFFTs = (nlocal / ng[2]);
                FFTs.backward(data,scratch,ng[2],nFFTs);

                dist.reorder(data,scratch,2,1);
                dist.returnPencils(data,scratch,2);
                dist.shuffle_indices(data,scratch,2);

                dist.getPencils(data,scratch,0);
                dist.reorder(data,scratch,0,0);

                nFFTs = (nlocal / ng[0]);
                FFTs.backward(data,scratch,ng[0],nFFTs);

                dist.reorder(data,scratch,0,1);
                dist.returnPencils(data,scratch,0);
                dist.shuffle_indices(data,scratch,0);

                dist.getPencils(data,scratch,1);
                dist.reorder(data,scratch,1,0);

                nFFTs = (nlocal / ng[1]);
                FFTs.backward(data,scratch,ng[1],nFFTs);

                dist.reorder(data,scratch,1,1);
                dist.returnPencils(data,scratch,1);
                dist.shuffle_indices(data,scratch,3);
            }

        }
        double end = MPI_Wtime();
        return end - start;
    }

    #ifdef GPU
    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
        return fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexFloatDevice* data, complexFloatDevice* scratch){
        return fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
        return fft(data,scratch,FFT_BACKWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexFloatDevice* data, complexFloatDevice* scratch){
        return fft(data,scratch,FFT_BACKWARD);
    }
    #endif

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexDoubleHost* data, complexDoubleHost* scratch){
        return fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexFloatHost* data, complexFloatHost* scratch){
        return fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexDoubleHost* data, complexDoubleHost* scratch){
        return fft(data,scratch,FFT_BACKWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    double Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexFloatHost* data, complexFloatHost* scratch){
        return fft(data,scratch,FFT_BACKWARD);
    }

    #ifdef FFTW
    template class Dfft<CPUMPI,CPUReorder,FFTWPlanManager>;
    #ifdef GPU
    template class Dfft<CPUMPI,GPUReorder,FFTWPlanManager>;
    #ifndef nocudampi
    template class Dfft<GPUMPI,CPUReorder,FFTWPlanManager>;
    template class Dfft<GPUMPI,GPUReorder,FFTWPlanManager>;
    #endif
    #endif
    #endif

    #ifdef GPU
    #ifdef GPUFFT
    template class Dfft<CPUMPI,CPUReorder,GPUPlanManager>;
    template class Dfft<CPUMPI,GPUReorder,GPUPlanManager>;
    #ifndef nocudampi
    template class Dfft<GPUMPI,CPUReorder,GPUPlanManager>;
    template class Dfft<GPUMPI,GPUReorder,GPUPlanManager>;
    #endif
    #endif
    #endif

}

#endif