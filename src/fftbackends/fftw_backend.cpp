#ifdef GPUFFT
#undef GPUFFT
#endif
#include "fftbackends/fftw_backend.hpp"
namespace SWFFT {
#ifdef SWFFT_FFTW

FFTWPlanManager::FFTWPlanManager() : m_last_size(0) {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        m_double_plans[i].valid = false;
        m_float_plans[i].valid = false;
    }
}

FFTWPlanManager::~FFTWPlanManager() {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        if (m_double_plans[i].valid) {
            fftw_destroy_plan(m_double_plans[i].plan);
        }
        if (m_float_plans[i].valid) {
            fftwf_destroy_plan(m_float_plans[i].plan);
        }
    }
    if (m_last_size != 0) {
        free(m_h_data);
        free(m_h_scratch);
    }
}

void FFTWPlanManager::query() { printf("Using fftw\n"); }

void FFTWPlanManager::allocate_host(size_t sz) {
    if (m_last_size == 0) {
        m_h_data = malloc(sz);
        m_h_scratch = malloc(sz);
        m_last_size = sz;
        return;
    }
    if (m_last_size < sz) {
        m_h_data = realloc(m_h_data, sz);
        m_h_scratch = realloc(m_h_scratch, sz);
        m_last_size = sz;
        return;
    }
    return;
}

fftw_plan FFTWPlanManager::find_plan(fftw_complex* data, fftw_complex* scratch,
                                     int ng, int nFFTs, int direction) {
    for (int i = 0; i < N_FFT_CACHE; i++) {
        if (m_double_plans[i].valid) {
            if ((m_double_plans[i].nFFTs == nFFTs) &&
                (m_double_plans[i].direction == direction) &&
                (m_double_plans[i].data == data) &&
                (m_double_plans[i].scratch == scratch) &&
                (m_double_plans[i].ng == ng)) {
                return m_double_plans[i].plan;
            }
        } else {
            m_double_plans[i].valid = true;
            int n[1] = {ng};
            m_double_plans[i].nFFTs = nFFTs;
            m_double_plans[i].ng = ng;
            m_double_plans[i].direction = direction;
            m_double_plans[i].data = data;
            m_double_plans[i].scratch = scratch;
            m_double_plans[i].plan =
                fftw_plan_many_dft(1, n, nFFTs, data, NULL, 1, ng, scratch,
                                   NULL, 1, ng, direction, FFTW_ESTIMATE);
            return m_double_plans[i].plan;
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
        if (m_float_plans[i].valid) {
            if ((m_float_plans[i].nFFTs == nFFTs) &&
                (m_float_plans[i].direction == direction) &&
                (m_float_plans[i].data == data) &&
                (m_float_plans[i].scratch == scratch) &&
                (m_float_plans[i].ng == ng)) {
                return m_float_plans[i].plan;
            }
        } else {
            m_float_plans[i].valid = true;
            int n[1] = {ng};
            m_float_plans[i].plan =
                fftwf_plan_many_dft(1, n, nFFTs, data, NULL, 1, ng, scratch,
                                    NULL, 1, ng, direction, FFTW_ESTIMATE);
            m_float_plans[i].nFFTs = nFFTs;
            m_float_plans[i].ng = ng;
            m_float_plans[i].direction = direction;
            m_float_plans[i].data = data;
            m_float_plans[i].scratch = scratch;
            return m_float_plans[i].plan;
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
    gpuMemcpy(m_h_data, data, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    forward((complexDoubleHost*)m_h_data, (complexDoubleHost*)m_h_scratch, ng,
            nFFTs);
    gpuMemcpy(scratch, m_h_scratch, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}

void FFTWPlanManager::forward(complexFloatDevice* data,
                              complexFloatDevice* scratch, int ng, int nFFTs) {
    allocate_host(sizeof(complexFloatHost) * ng * nFFTs);
    gpuMemcpy(m_h_data, data, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    forward((complexFloatHost*)m_h_data, (complexFloatHost*)m_h_scratch, ng,
            nFFTs);
    gpuMemcpy(scratch, m_h_scratch, sizeof(complexFloatHost) * ng * nFFTs,
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
    gpuMemcpy(m_h_data, data, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    backward((complexDoubleHost*)m_h_data, (complexDoubleHost*)m_h_scratch, ng,
             nFFTs);
    gpuMemcpy(scratch, m_h_scratch, sizeof(complexDoubleHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}

void FFTWPlanManager::backward(complexFloatDevice* data,
                               complexFloatDevice* scratch, int ng, int nFFTs) {
    allocate_host(sizeof(complexFloatHost) * ng * nFFTs);
    gpuMemcpy(m_h_data, data, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyDeviceToHost);
    backward((complexFloatHost*)m_h_data, (complexFloatHost*)m_h_scratch, ng,
             nFFTs);
    gpuMemcpy(scratch, m_h_scratch, sizeof(complexFloatHost) * ng * nFFTs,
              gpuMemcpyHostToDevice);
}
#endif // SWFFT_GPU

#endif // SWFFT_FFTW
} // namespace SWFFT