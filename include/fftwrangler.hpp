#ifndef FFT_WRANGLER_INCLUDED
#define FFT_WRANGLER_INCLUDED

#define N_FFT_CACHE 100

#include "complex-type.h"

//#ifdef GPU
#include "gpu.hpp"
//#endif

#ifdef FFTW
#include <fftw3.h>
#include <map>

template<class T, class plan_t>
class FFTWPlanWrapper{
    public:
        int nFFTs;
        T* data;
        T* scratch;
        plan_t plan;
        int direction;
        int ng;
        bool valid;
};

class FFTWPlanManager{
    public:
        FFTWPlanWrapper<fftw_complex,fftw_plan> double_plans[N_FFT_CACHE];
        FFTWPlanWrapper<fftwf_complex,fftwf_plan> float_plans[N_FFT_CACHE];
        
        FFTWPlanManager();
        ~FFTWPlanManager();

        fftw_plan find_plan(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs, int direction);
        fftwf_plan find_plan(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs, int direction);

        void forward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs);
        void forward(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs);

        void forward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs);
        void forward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs);

        void forward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs);
        void forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs);

        void backward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs);
        void backward(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs);

        void backward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs);
        void backward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs);
        
        void backward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs);
        void backward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs);

};
#endif

#ifdef GPU

class GPUPlanWrapper{
    public:
        int nFFTs;
        gpufftHandle plan;
        gpufftType t;
        int ng;
        bool valid;
};

class GPUPlanManager{
    public:
        GPUPlanWrapper plans[N_FFT_CACHE];

        GPUPlanManager();
        ~GPUPlanManager();

        gpufftHandle find_plan(int ng, int nFFTs, gpufftType t);
        
        void forward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs);
        void forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs);

        void forward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs);
        void forward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs);

        void backward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs);
        void backward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs);

        void backward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs);
        void backward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs);
};

#endif

#endif