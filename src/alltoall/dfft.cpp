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
void Dfft<T, FFTBackend>::makePlans(T* data_, T* scratch_){

    int nFFTs = (nlocal / distribution.batches) / Ng;
    FFTs.cachePlans(data_,scratch_,Ng,nFFTs,FFT_FORWARD);
    FFTs.cachePlans(data_,scratch_,Ng,nFFTs,FFT_BACKWARD);
}

#ifdef GPU
template class Dfft<complexDouble,GPUFFT>;
template class Dfft<complexFloat,GPUFFT>;
#endif

//void makePlans(T* scratch_);
//void makePlans();

}


#endif