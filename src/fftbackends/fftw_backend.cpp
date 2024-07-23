#ifdef GPUFFT
#undef GPUFFT
#endif
#include "fftbackends/fftw_backend.hpp"
namespace SWFFT {
#ifdef SWFFT_FFTW

FFTWPlanManager::FFTWPlanManager() : last_size(0) {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        double_plans[i].valid = false;
        float_plans[i].valid = false;
    }
}

FFTWPlanManager::~FFTWPlanManager() {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        if (double_plans[i].valid) {
            fftw_destroy_plan(double_plans[i].plan);
        }
        if (float_plans[i].valid) {
            fftwf_destroy_plan(float_plans[i].plan);
        }
    }
    if (last_size != 0) {
        free(h_data);
        free(h_scratch);
    }
}

void FFTWPlanManager::query() { printf("Using fftw\n"); }

void FFTWPlanManager::allocate_host(size_t sz) {
    if (last_size == 0) {
        h_data = malloc(sz);
        h_scratch = malloc(sz);
        last_size = sz;
        return;
    }
    if (last_size < sz) {
        h_data = realloc(h_data, sz);
        h_scratch = realloc(h_scratch, sz);
        last_size = sz;
        return;
    }
    return;
}

fftw_plan FFTWPlanManager::find_plan(fftw_complex* data, fftw_complex* scratch,
                                     int ng, int nFFTs, int direction) {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        if (double_plans[i].valid) {
            if ((double_plans[i].nFFTs == nFFTs) &&
                (double_plans[i].direction == direction) &&
                (double_plans[i].data == data) &&
                (double_plans[i].scratch == scratch) &&
                (double_plans[i].ng == ng)) {
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
            double_plans[i].plan =
                fftw_plan_many_dft(1, n, nFFTs, data, NULL, 1, ng, scratch,
                                   NULL, 1, ng, direction, FFTW_ESTIMATE);
            return double_plans[i].plan;
        }
    }
    int n[1] = {ng};
    return fftw_plan_many_dft(1, n, nFFTs, data, NULL, 1, ng, scratch, NULL, 1,
                              ng, direction, FFTW_ESTIMATE);
}

fftwf_plan FFTWPlanManager::find_plan(fftwf_complex* data,
                                      fftwf_complex* scratch, int ng, int nFFTs,
                                      int direction) {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        if (float_plans[i].valid) {
            if ((float_plans[i].nFFTs == nFFTs) &&
                (float_plans[i].direction == direction) &&
                (float_plans[i].data == data) &&
                (float_plans[i].scratch == scratch) &&
                (float_plans[i].ng == ng)) {
                return float_plans[i].plan;
            }
        } else {
            float_plans[i].valid = true;
            int n[1] = {ng};
            float_plans[i].plan =
                fftwf_plan_many_dft(1, n, nFFTs, data, NULL, 1, ng, scratch,
                                    NULL, 1, ng, direction, FFTW_ESTIMATE);
            float_plans[i].nFFTs = nFFTs;
            float_plans[i].ng = ng;
            float_plans[i].direction = direction;
            float_plans[i].data = data;
            float_plans[i].scratch = scratch;
            return float_plans[i].plan;
        }
    }
    int n[1] = {ng};
    return fftwf_plan_many_dft(1, n, nFFTs, data, NULL, 1, ng, scratch, NULL, 1,
                               ng, direction, FFTW_ESTIMATE);
}

void FFTWPlanManager::forward(fftw_complex* data, fftw_complex* scratch, int ng,
                              int nFFTs) {
    fftw_execute(find_plan(data, scratch, ng, nFFTs, FFTW_FORWARD));
}

void FFTWPlanManager::forward(fftwf_complex* data, fftwf_complex* scratch,
                              int ng, int nFFTs) {
    fftwf_execute(find_plan(data, scratch, ng, nFFTs, FFTW_FORWARD));
}

void FFTWPlanManager::forward(complexDoubleHost* data,
                              complexDoubleHost* scratch, int ng, int nFFTs) {
    forward((fftw_complex*)data, (fftw_complex*)scratch, ng, nFFTs);
}

void FFTWPlanManager::forward(complexFloatHost* data, complexFloatHost* scratch,
                              int ng, int nFFTs) {
    forward((fftwf_complex*)data, (fftwf_complex*)scratch, ng, nFFTs);
}

#ifdef SWFFT_GPU
void FFTWPlanManager::forward(complexDoubleDevice* data,
                              complexDoubleDevice* scratch, int ng, int nFFTs) {
    allocate_host(sizeof(complexDoubleHost) * ng * nFFTs);
    gpuMemcpy(h_data, data, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    forward((complexDoubleHost*)h_data, (complexDoubleHost*)h_scratch, ng,
            nFFTs);
    gpuMemcpy(scratch, h_scratch, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}

void FFTWPlanManager::forward(complexFloatDevice* data,
                              complexFloatDevice* scratch, int ng, int nFFTs) {
    allocate_host(sizeof(complexFloatHost) * ng * nFFTs);
    gpuMemcpy(h_data, data, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    forward((complexFloatHost*)h_data, (complexFloatHost*)h_scratch, ng, nFFTs);
    gpuMemcpy(scratch, h_scratch, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}
#endif // SWFFT_GPU

void FFTWPlanManager::backward(fftw_complex* data, fftw_complex* scratch,
                               int ng, int nFFTs) {
    fftw_execute(find_plan(data, scratch, ng, nFFTs, FFTW_BACKWARD));
}

void FFTWPlanManager::backward(fftwf_complex* data, fftwf_complex* scratch,
                               int ng, int nFFTs) {
    fftwf_execute(find_plan(data, scratch, ng, nFFTs, FFTW_BACKWARD));
}

void FFTWPlanManager::backward(complexDoubleHost* data,
                               complexDoubleHost* scratch, int ng, int nFFTs) {
    backward((fftw_complex*)data, (fftw_complex*)scratch, ng, nFFTs);
}

void FFTWPlanManager::backward(complexFloatHost* data,
                               complexFloatHost* scratch, int ng, int nFFTs) {
    backward((fftwf_complex*)data, (fftwf_complex*)scratch, ng, nFFTs);
}

#ifdef SWFFT_GPU
void FFTWPlanManager::backward(complexDoubleDevice* data,
                               complexDoubleDevice* scratch, int ng,
                               int nFFTs) {
    allocate_host(sizeof(complexDoubleHost) * ng * nFFTs);
    gpuMemcpy(h_data, data, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    backward((complexDoubleHost*)h_data, (complexDoubleHost*)h_scratch, ng,
             nFFTs);
    gpuMemcpy(scratch, h_scratch, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}

void FFTWPlanManager::backward(complexFloatDevice* data,
                               complexFloatDevice* scratch, int ng, int nFFTs) {
    allocate_host(sizeof(complexFloatHost) * ng * nFFTs);
    gpuMemcpy(h_data, data, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    backward((complexFloatHost*)h_data, (complexFloatHost*)h_scratch, ng,
             nFFTs);
    gpuMemcpy(scratch, h_scratch, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}
#endif // SWFFT_GPU

#endif // SWFFT_FFTW
} // namespace SWFFT