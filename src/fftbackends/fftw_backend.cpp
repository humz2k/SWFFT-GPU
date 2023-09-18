#ifdef GPUFFT
#undef GPUFFT
#endif
#include "fftwrangler.hpp"

#ifdef FFTW

FFTWPlanManager::FFTWPlanManager(){
    for (int i = 0; i < N_FFT_CACHE; i++){
        double_plans[i].valid = false;
        float_plans[i].valid = false;
    }
}

FFTWPlanManager::~FFTWPlanManager(){
    for (int i = 0; i < N_FFT_CACHE; i++){
        if (double_plans[i].valid){
            fftw_destroy_plan(double_plans[i].plan);
        }
        if (float_plans[i].valid){
            fftwf_destroy_plan(float_plans[i].plan);
        }
    }
}


fftw_plan FFTWPlanManager::find_plan(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs, int direction){
    for (int i = 0; i < N_FFT_CACHE; i++){
        if (double_plans[i].valid){
            if ((double_plans[i].nFFTs == nFFTs) && 
                (double_plans[i].direction == direction) && 
                (double_plans[i].data == data) &&
                (double_plans[i].scratch == scratch) &&
                (double_plans[i].ng == ng)){
                //printf("FOUND CACHED PLAN!\n");
                return double_plans[i].plan;
            }
        } else {
            double_plans[i].valid = true;
            int n[1] = {ng};
            double_plans[i].nFFTs = nFFTs;
            double_plans[i].ng = ng;
            double_plans[i].direction = direction;
            double_plans[i].data = data;
            double_plans[i].scratch = scratch;
            double_plans[i].plan = fftw_plan_many_dft(1,n,nFFTs,data,NULL,1,ng,scratch,NULL,1,ng,direction, FFTW_ESTIMATE);
            //printf("DIDNT FIND CACHED PLAN, ADDING TO CACHE!\n");
            return double_plans[i].plan;
        }
    }
    int n[1] = {ng};
    //printf("OUT OF CACHE SPACE!!!\n");
    return fftw_plan_many_dft(1,n,nFFTs,data,NULL,1,ng,scratch,NULL,1,ng,direction,FFTW_ESTIMATE);
}


fftwf_plan FFTWPlanManager::find_plan(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs, int direction){
    for (int i = 0; i < N_FFT_CACHE; i++){
        if (float_plans[i].valid){
            if ((float_plans[i].nFFTs == nFFTs) && 
                (float_plans[i].direction == direction) && 
                (float_plans[i].data == data) &&
                (float_plans[i].scratch == scratch) &&
                (float_plans[i].ng == ng)){
                return float_plans[i].plan;
            }
        } else {
            float_plans[i].valid = true;
            int n[1] = {ng};
            float_plans[i].plan = fftwf_plan_many_dft(1,n,nFFTs,data,NULL,1,ng,scratch,NULL,1,ng,direction,FFTW_ESTIMATE);
            float_plans[i].nFFTs = nFFTs;
            float_plans[i].ng = ng;
            float_plans[i].direction = direction;
            float_plans[i].data = data;
            float_plans[i].scratch = scratch;
            return float_plans[i].plan;
        }
    }
    int n[1] = {ng};
    return fftwf_plan_many_dft(1,n,nFFTs,data,NULL,1,ng,scratch,NULL,1,ng,direction,FFTW_ESTIMATE);
}

void FFTWPlanManager::forward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs){
    fftw_execute(find_plan(data,scratch,ng,nFFTs,FFTW_FORWARD));
}

void FFTWPlanManager::forward(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs){
    fftwf_execute(find_plan(data,scratch,ng,nFFTs,FFTW_FORWARD));
}

void FFTWPlanManager::forward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs){
    forward((fftw_complex*)data,(fftw_complex*)scratch,ng,nFFTs);
}

void FFTWPlanManager::forward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs){
    forward((fftwf_complex*)data,(fftwf_complex*)scratch,ng,nFFTs);
}

#ifdef GPU
void FFTWPlanManager::forward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs){
    complexDoubleHost* h_data; swfftAlloc(&h_data,sizeof(complexDoubleHost) * ng * nFFTs);
    complexDoubleHost* h_scratch; swfftAlloc(&h_scratch,sizeof(complexDoubleHost) * ng * nFFTs);
    gpuMemcpy(h_data,data,sizeof(complexDoubleHost) * ng * nFFTs, gpuMemcpyDeviceToHost);
    forward(h_data,h_scratch,ng,nFFTs);
    gpuMemcpy(scratch,h_scratch,sizeof(complexDoubleHost) * ng * nFFTs, gpuMemcpyHostToDevice);
}

void FFTWPlanManager::forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs){
    complexFloatHost* h_data; swfftAlloc(&h_data,sizeof(complexFloatHost) * ng * nFFTs);
    complexFloatHost* h_scratch; swfftAlloc(&h_scratch,sizeof(complexFloatHost) * ng * nFFTs);
    gpuMemcpy(h_data,data,sizeof(complexFloatHost) * ng * nFFTs, gpuMemcpyDeviceToHost);
    forward(h_data,h_scratch,ng,nFFTs);
    gpuMemcpy(scratch,h_scratch,sizeof(complexFloatHost) * ng * nFFTs, gpuMemcpyHostToDevice);
}
#endif

void FFTWPlanManager::backward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs){
    fftw_execute(find_plan(data,scratch,ng,nFFTs,FFTW_BACKWARD));
}

void FFTWPlanManager::backward(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs){
    fftwf_execute(find_plan(data,scratch,ng,nFFTs,FFTW_BACKWARD));
}

void FFTWPlanManager::backward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs){
    backward((fftw_complex*)data,(fftw_complex*)scratch,ng,nFFTs);
}

void FFTWPlanManager::backward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs){
    backward((fftwf_complex*)data,(fftwf_complex*)scratch,ng,nFFTs);
}

#ifdef GPU
void FFTWPlanManager::backward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs){
    complexDoubleHost* h_data; swfftAlloc(&h_data,sizeof(complexDoubleHost) * ng * nFFTs);
    complexDoubleHost* h_scratch; swfftAlloc(&h_scratch,sizeof(complexDoubleHost) * ng * nFFTs);
    gpuMemcpy(h_data,data,sizeof(complexDoubleHost) * ng * nFFTs, gpuMemcpyDeviceToHost);
    backward(h_data,h_scratch,ng,nFFTs);
    gpuMemcpy(scratch,h_scratch,sizeof(complexDoubleHost) * ng * nFFTs, gpuMemcpyHostToDevice);
}

void FFTWPlanManager::backward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs){
    complexFloatHost* h_data; swfftAlloc(&h_data,sizeof(complexFloatHost) * ng * nFFTs);
    complexFloatHost* h_scratch; swfftAlloc(&h_scratch,sizeof(complexFloatHost) * ng * nFFTs);
    gpuMemcpy(h_data,data,sizeof(complexFloatHost) * ng * nFFTs, gpuMemcpyDeviceToHost);
    backward(h_data,h_scratch,ng,nFFTs);
    gpuMemcpy(scratch,h_scratch,sizeof(complexFloatHost) * ng * nFFTs, gpuMemcpyHostToDevice);
}
#endif

#endif
