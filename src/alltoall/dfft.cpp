#include "alltoall.hpp"

#ifdef ALLTOALL
namespace A2A{

template<class T, template<class> class FFTBackend>
Dfft<T, FFTBackend>::Dfft(Distribution<T> &dist) : distribution(dist){

    Ng = distribution.Ng;
    world_size = distribution.world_size;
    nlocal = distribution.nlocal;

    #ifdef GPU
    blockSize = distribution.blockSize;
    gpuStreamCreate(&fftstream);
    fft_events = (gpuEvent_t*)malloc(sizeof(gpuEvent_t) * distribution.batches);
    for (int i = 0; i < distribution.batches; i++){
        gpuEventCreate(&fft_events[i]);
    }
    #endif

    PlansMade = 0;

}

template<class T, template<class> class FFTBackend>
Dfft<T, FFTBackend>::~Dfft(){

    finalize();

}

template<class T, template<class> class FFTBackend>
void Dfft<T, FFTBackend>::makePlans(T* data_, T* scratch_){

    if(PlansMade == 0){
        int nFFTs = (nlocal / distribution.batches) / Ng;
        FFTs.cachePlans(data_,scratch_,Ng,nFFTs,FFT_FORWARD,fftstream);
        FFTs.cachePlans(data_,scratch_,Ng,nFFTs,FFT_BACKWARD,fftstream);
        data = data_;
        scratch = scratch_;
        PlansMade = 1;
    }

}

template<class T, template<class> class FFTBackend>
void Dfft<T, FFTBackend>::makePlans(T* scratch_){

    if(PlansMade == 0){

        int nFFTs = (nlocal / distribution.batches) / Ng;
        FFTs.cachePlans(scratch_,Ng,nFFTs,FFT_FORWARD,fftstream);
        FFTs.cachePlans(scratch_,Ng,nFFTs,FFT_BACKWARD,fftstream);
        scratch = scratch_;
        PlansMade = 2;
    }

}

/*template<class T, template<class> class FFTBackend>
void Dfft<T, FFTBackend>::makePlans(){

    if(PlansMade == 0){

        int nFFTs = (nlocal / distribution.batches) / Ng;
        FFTs.cachePlans(Ng,nFFTs,FFT_FORWARD);
        FFTs.cachePlans(Ng,nFFTs,FFT_BACKWARD);
        PlansMade = 3;
    }

}*/

template<class T, template<class> class FFTBackend>
void Dfft<T,FFTBackend>::finalize(){
    gpuStreamDestroy(fftstream);
    for (int i = 0; i < distribution.batches; i++){
        gpuEventDestroy(fft_events[i]);
    }
    free(fft_events);
}


template<class T, template<class> class FFTBackend>
inline void Dfft<T, FFTBackend>::fft(T* data, fftdirection direction)
{
    #pragma GCC unroll 3
    for (int i = 0; i < 3; i++){
        #ifdef nocudampi
        for (int batch = 0; batch < distribution.batches; batch++){
            distribution.memcpy_d2h(distribution.h_scratch1,data,batch,distribution.diststream);
            gpuEventRecord(fft_events[batch],distribution.diststream);
        }
        #endif

        for (int batch = 0; batch < distribution.batches; batch++){
            #ifdef nocudampi

            gpuEventSynchronize(fft_events[batch]);
            distribution.getPencils((T*)distribution.h_scratch1,(T*)distribution.h_scratch2,i,batch);
            distribution.memcpy_h2d(scratch,distribution.h_scratch2,batch,fftstream);
            #else
            distribution.getPencils(data,scratch,i,batch);
            #endif

            distribution.reorder(data,scratch,i,0,batch,fftstream);
            int b_start = batch * (distribution.nlocal / distribution.batches);
            int nFFTs = (nlocal / distribution.batches) / Ng;
            #ifdef customalltoall
            //FFTs.fft(&data[b_start],&scratch[b_start],ng,)
            #else
            FFTs.fft(&data[b_start],&scratch[b_start],Ng,nFFTs,direction,fftstream);
            #endif
            
            distribution.reorder(data,scratch,i,1,batch,fftstream);
            #ifdef nocudampi
            distribution.memcpy_d2h(distribution.h_scratch1,data,batch,fftstream);
            #endif
            gpuEventRecord(fft_events[batch],fftstream);
        }

        for (int batch = 0; batch < distribution.batches; batch++){
            gpuEventSynchronize(fft_events[batch]);
            #ifdef nocudampi

            distribution.returnPencils((T*)distribution.h_scratch1,(T*)distribution.h_scratch2,i,batch);

            distribution.memcpy_h2d(scratch,distribution.h_scratch2,batch,distribution.diststream);
            
            #else
            distribution.returnPencils(data,scratch,i,batch);
            #endif
        }
        #ifdef nocudampi
        gpuStreamSynchronize(distribution.diststream);
        #endif

        distribution.shuffle_indices(data,scratch,i);
    }

    #ifdef verbose
    if (distribution.world_rank == 0){printf("Finished 3D FFT\n");}
    #endif
}

template<class T, template<class> class FFTBackend>
void Dfft<T,FFTBackend>::forward(T* data_)
{
    fft(data_,FFT_FORWARD);   
}

template<class T, template<class> class FFTBackend>
void Dfft<T,FFTBackend>::backward(T* data_){
    fft(data_,FFT_BACKWARD);
}

template<class T, template<class> class FFTBackend>
void Dfft<T,FFTBackend>::forward()
{
    fft(data,FFT_FORWARD);   
}

template<class T, template<class> class FFTBackend>
void Dfft<T,FFTBackend>::backward(){
    fft(data,FFT_BACKWARD);
}



#ifdef GPU
template class Dfft<complexDouble,GPUFFT>;
template class Dfft<complexFloat,GPUFFT>;
#endif

//void makePlans(T* scratch_);
//void makePlans();

}


#endif