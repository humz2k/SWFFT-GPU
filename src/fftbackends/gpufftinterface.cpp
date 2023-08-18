#include "fftinterface.hpp"

#ifdef FFTW
#undef FFTW
#endif

#ifdef GPU

template<class T>
GPUFFT<T>::GPUFFT(){
    printf("Using GPU FFTs!\n");
    for (int i = 0; i < nplans; i++){
        ns[i] = 0;
        ngs[i] = 0;
    }
}

template<class T>
GPUFFT<T>::~GPUFFT(){
    printf("Destorying GPU FFT Interface!\n");
}

template<>
gpufftHandle GPUFFT<complexDouble>::findPlans(int ng, int nFFTs){
    for (int i = 0; i < nplans; i++){
        if ((ns[i] == nFFTs) && (ngs[i] == ng)){
            printf("Found cached plan!\n");
            return plans[i];
        }
        if (ns[i] == 0){
            if (gpufftPlan1d(&plans[i], ng, GPUFFT_Z2Z, nFFTs) != GPUFFT_SUCCESS){
                printf("CUFFT error: Plan creation failed\n");
                exit(1);
            }
            return plans[i];
        }
    }
    printf("Out of space for plans!\n");
    exit(1);
}

template<>
gpufftHandle GPUFFT<complexFloat>::findPlans(int ng, int nFFTs){
    for (int i = 0; i < nplans; i++){
        if ((ns[i] == nFFTs) && (ngs[i] == ng)){
            printf("Found cached plan!\n");
            return plans[i];
        }
        if (ns[i] == 0){
            if (gpufftPlan1d(&plans[i], ng, GPUFFT_C2C, nFFTs) != GPUFFT_SUCCESS){
                printf("CUFFT error: Plan creation failed\n");
                exit(1);
            }
            return plans[i];
        }
    }
    printf("Out of space for plans!\n");
    exit(1);
}

template<>
gpufftHandle GPUFFT<complexDouble>::findPlans(int ng, int nFFTs,gpuStream_t stream){
    for (int i = 0; i < nplans; i++){
        if ((ns[i] == nFFTs) && (ngs[i] == ng)){
            printf("Found cached plan!\n");
            return plans[i];
        }
        if (ns[i] == 0){
            if (gpufftPlan1d(&plans[i], ng, GPUFFT_Z2Z, nFFTs) != GPUFFT_SUCCESS){
                printf("CUFFT error: Plan creation failed\n");
                exit(1);
            }
            return plans[i];
        }
    }
    printf("Out of space for plans!\n");
    exit(1);
}

template<>
gpufftHandle GPUFFT<complexFloat>::findPlans(int ng, int nFFTs, gpuStream_t stream){
    for (int i = 0; i < nplans; i++){
        if ((ns[i] == nFFTs) && (ngs[i] == ng)){
            printf("Found cached plan!\n");
            return plans[i];
        }
        if (ns[i] == 0){
            if (gpufftPlan1d(&plans[i], ng, GPUFFT_C2C, nFFTs) != GPUFFT_SUCCESS){
                printf("CUFFT error: Plan creation failed\n");
                exit(1);
            }
            return plans[i];
        }
    }
    printf("Out of space for plans!\n");
    exit(1);
}

template<class T>
void GPUFFT<T>::cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction){
    findPlans(ng,nFFTs);
}

template<class T>
void GPUFFT<T>::cachePlans(T* scratch, int ng, int nFFTs, fftdirection direction){
    findPlans(ng,nFFTs);
}

template<class T>
void GPUFFT<T>::cachePlans(T* data, T* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream){
    findPlans(ng,nFFTs,stream);
}

template<class T>
void GPUFFT<T>::cachePlans(T* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream){
    findPlans(ng,nFFTs,stream);
}

template<>
void GPUFFT<complexDouble>::fft(complexDouble* data, complexDouble* scratch, int ng, int nFFTs, fftdirection direction){
    gpufftHandle plan = findPlans(ng,nFFTs);
    int dir = GPUFFT_INVERSE;
    if (direction == FFT_FORWARD)dir = GPUFFT_FORWARD;
    if (gpufftExecZ2Z(plan, data, scratch, dir) != GPUFFT_SUCCESS){
        char backstr[] = "Backward";
        char forstr[] = "Forward";
        char* dirstr = backstr;
        if (dir == GPUFFT_FORWARD){
            dirstr = forstr;
        }
        printf("CUFFT error: ExecZ2Z %s failed\n",dirstr);
        return;	
    }
}

template<>
void GPUFFT<complexFloat>::fft(complexFloat* data, complexFloat* scratch, int ng, int nFFTs, fftdirection direction){
    gpufftHandle plan = findPlans(ng,nFFTs);
    int dir = GPUFFT_INVERSE;
    if (direction == FFT_FORWARD)dir = GPUFFT_FORWARD;
    if (gpufftExecC2C(plan, data, scratch, dir) != GPUFFT_SUCCESS){
        char backstr[] = "Backward";
        char forstr[] = "Forward";
        char* dirstr = backstr;
        if (dir == GPUFFT_FORWARD){
            dirstr = forstr;
        }
        printf("CUFFT error: ExecZ2Z %s failed\n",dirstr);
        return;	
    }
}

template<>
void GPUFFT<complexDouble>::fft(complexDouble* data, complexDouble* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream){
    gpufftHandle plan = findPlans(ng,nFFTs);
    int dir = GPUFFT_INVERSE;
    if (direction == FFT_FORWARD)dir = GPUFFT_FORWARD;
    if (gpufftExecZ2Z(plan, data, scratch, dir) != GPUFFT_SUCCESS){
        char backstr[] = "Backward";
        char forstr[] = "Forward";
        char* dirstr = backstr;
        if (dir == GPUFFT_FORWARD){
            dirstr = forstr;
        }
        printf("CUFFT error: ExecZ2Z %s failed\n",dirstr);
        return;	
    }
}

template<>
void GPUFFT<complexFloat>::fft(complexFloat* data, complexFloat* scratch, int ng, int nFFTs, fftdirection direction, gpuStream_t stream){
    gpufftHandle plan = findPlans(ng,nFFTs);
    int dir = GPUFFT_INVERSE;
    if (direction == FFT_FORWARD)dir = GPUFFT_FORWARD;
    if (gpufftExecC2C(plan, data, scratch, dir) != GPUFFT_SUCCESS){
        char backstr[] = "Backward";
        char forstr[] = "Forward";
        char* dirstr = backstr;
        if (dir == GPUFFT_FORWARD){
            dirstr = forstr;
        }
        printf("CUFFT error: ExecZ2Z %s failed\n",dirstr);
        return;	
    }
}

template class GPUFFT<complexFloat>;
template class GPUFFT<complexDouble>;
#endif