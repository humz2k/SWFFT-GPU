#ifdef ALLTOALL
#include "alltoall.hpp"

namespace A2A{
    template<class MPI_T,class REORDER_T,class FFTBackend>
    Dfft<MPI_T,REORDER_T,FFTBackend>::Dfft(Distribution<MPI_T,REORDER_T>& dist_) : dist(dist_){
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
    template<class T>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::fft(T* data, T* scratch, fftdirection direction){

        #pragma GCC unroll 3
        for (int i = 0; i < 3; i++){
            dist.getPencils(data,scratch,i);
            dist.reorder(data,scratch,i,0);

            int nFFTs = (nlocal / ng[i]);
            if (direction == FFT_FORWARD){
                FFTs.forward(data,scratch,ng[i],nFFTs);
            } else {
                FFTs.backward(data,scratch,ng[i],nFFTs);
            }

            dist.reorder(data,scratch,i,1);

            dist.returnPencils(data,scratch,i);

            dist.shuffle_indices(data,scratch,i);

        }

    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
        fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexDoubleHost* data, complexDoubleHost* scratch){
        fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexFloatDevice* data, complexFloatDevice* scratch){
        fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::forward(complexFloatHost* data, complexFloatHost* scratch){
        fft(data,scratch,FFT_FORWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
        fft(data,scratch,FFT_BACKWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexDoubleHost* data, complexDoubleHost* scratch){
        fft(data,scratch,FFT_BACKWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexFloatDevice* data, complexFloatDevice* scratch){
        fft(data,scratch,FFT_BACKWARD);
    }

    template<class MPI_T,class REORDER_T,class FFTBackend>
    void Dfft<MPI_T,REORDER_T,FFTBackend>::backward(complexFloatHost* data, complexFloatHost* scratch){
        fft(data,scratch,FFT_BACKWARD);
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