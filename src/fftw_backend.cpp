#ifdef GPU
#undef GPU
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
            double_plans[i].plan = fftw_plan_many_dft(1,n,nFFTs,reinterpret_cast<double(*)[2]>(data),NULL,1,ng,reinterpret_cast<double(*)[2]>(scratch),NULL,1,ng,direction,FFTW_MEASURE);
            double_plans[i].nFFTs = nFFTs;
            double_plans[i].ng = ng;
            double_plans[i].direction = direction;
            double_plans[i].data = data;
            double_plans[i].scratch = scratch;
            //printf("DIDNT FIND CACHED PLAN, ADDING TO CACHE!\n");
            return double_plans[i].plan;
        }
    }
    int n[1] = {ng};
    //printf("OUT OF CACHE SPACE!!!\n");
    return fftw_plan_many_dft(1,n,nFFTs,reinterpret_cast<double(*)[2]>(data),NULL,1,ng,reinterpret_cast<double(*)[2]>(scratch),NULL,1,ng,direction,FFTW_MEASURE);
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
            float_plans[i].plan = fftwf_plan_many_dft(1,n,nFFTs,reinterpret_cast<float(*)[2]>(data),NULL,1,ng,reinterpret_cast<float(*)[2]>(scratch),NULL,1,ng,direction,FFTW_MEASURE);
            float_plans[i].nFFTs = nFFTs;
            float_plans[i].ng = ng;
            float_plans[i].direction = direction;
            float_plans[i].data = data;
            float_plans[i].scratch = scratch;
            return float_plans[i].plan;
        }
    }
    int n[1] = {ng};
    return fftwf_plan_many_dft(1,n,nFFTs,reinterpret_cast<float(*)[2]>(data),NULL,1,ng,reinterpret_cast<float(*)[2]>(scratch),NULL,1,ng,direction,FFTW_MEASURE);
}

void FFTWPlanManager::forward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs){
    fftw_execute(find_plan(data,scratch,ng,nFFTs,FFTW_FORWARD));
}

void FFTWPlanManager::forward(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs){
    fftwf_execute(find_plan(data,scratch,ng,nFFTs,FFTW_FORWARD));
}

void FFTWPlanManager::backward(fftw_complex* data, fftw_complex* scratch, int ng, int nFFTs){
    fftw_execute(find_plan(data,scratch,ng,nFFTs,FFTW_BACKWARD));
}

void FFTWPlanManager::backward(fftwf_complex* data, fftwf_complex* scratch, int ng, int nFFTs){
    fftwf_execute(find_plan(data,scratch,ng,nFFTs,FFTW_BACKWARD));
}

#endif
