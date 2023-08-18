#ifndef FFTSEEN
#define FFTSEEN

#include <stdio.h>
#include <stdlib.h>
#include "complex-type.h"

#define nplans 10

enum fftdirection {FFT_FORWARD, FFT_BACKWARD};

template<class T>
class FFTInterface{
    public:
        FFTInterface();
        ~FFTInterface();

        void cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);

        void cachePlans(T* scratch, int ng, int nFFTs, fftdirection direction);

        //void cachePlans(int ng, int nFFTs, fftdirection direction);

        void fft(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);
};

#ifdef GPU
#include <gpu.hpp>
template<class T>
class GPUFFT{
    public:

        gpufftHandle plans[nplans];
        int ns[nplans];
        int ngs[nplans];

        GPUFFT();
        ~GPUFFT();

        gpufftHandle findPlans(int ng, int nFFTs);

        gpufftHandle findPlans(int ng, int nFFTs, gpuStream_t stream);

        void cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);

        void cachePlans(T* scratch, int ng, int nFFTs, fftdirection direction);

        void cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream);

        void cachePlans(T* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream);

        void fft(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);

        void fft(T* data, T* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream);

};

#endif

#ifdef FFTW

#endif
#endif