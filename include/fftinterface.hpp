#ifndef FFTSEEN
#define FFTSEEN

#include <stdio.h>
#include <stdlib.h>

#define nplans 10

enum fftdirection {FFT_FORWARD, FFT_BACKWARD};

template<class T>
class FFTInterface{
    public:
        FFTInterface();
        ~FFTInterface();

        void cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);

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

        void cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);

        void fft(T* data, T* scratch, int ng, int nFFTs, fftdirection direction);

};

#endif

#ifdef FFTW

#endif
#endif