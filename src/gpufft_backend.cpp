#ifdef FFTW
#undef FFTW
#endif
#include "fftwrangler.hpp"

#ifdef GPU

#include <stdio.h>
#include <stdlib.h>

GPUPlanManager::GPUPlanManager(){
    for (int i = 0; i < N_FFT_CACHE; i++){
        plans[i].valid = false;
    }
}

GPUPlanManager::~GPUPlanManager(){
    for (int i = 0; i < N_FFT_CACHE; i++){
        if (plans[i].valid){
            gpufftDestroy(plans[i].plan);
        }
    }
}

gpufftHandle GPUPlanManager::find_plan(int ng, int nFFTs, gpufftType t){
    for (int i = 0; i < N_FFT_CACHE; i++){
        if (plans[i].valid){
            if ((plans[i].ng == ng) && (plans[i].nFFTs == nFFTs) && (plans[i].t == t)){
                return plans[i].plan;
            }
        } else {
            plans[i].valid = true;
            plans[i].ng = ng;
            plans[i].nFFTs = nFFTs;
            plans[i].t = t;
            if (gpufftPlan1d(&plans[i].plan,ng,t,nFFTs) != GPUFFT_SUCCESS){
                printf("CUFFT error: Plan creation failed\n");
                exit(1);
            }
            return plans[i].plan;

        }
    }
    printf("Out of space for plans!\n");
    exit(1);
}

void GPUPlanManager::forward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs){
    if (gpufftExecZ2Z(find_plan(ng,nFFTs,GPUFFT_Z2Z),data,scratch,GPUFFT_FORWARD) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z FORWARD failed\n");
        return;
    }
    gpuDeviceSynchronize();
}

void GPUPlanManager::forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs){
    if (gpufftExecC2C(find_plan(ng,nFFTs,GPUFFT_C2C),data,scratch,GPUFFT_FORWARD) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C FORWARD failed\n");
        return;
    }
    gpuDeviceSynchronize();
}

void GPUPlanManager::forward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs){
    complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * ng * nFFTs);
    complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * ng * nFFTs);
    gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * ng * nFFTs,gpuMemcpyHostToDevice);
    forward(d_data,d_scratch,ng,nFFTs);
    gpuMemcpy(scratch,d_scratch,sizeof(complexDoubleDevice) * ng * nFFTs,gpuMemcpyDeviceToHost);
    swfftFree(d_data);
    swfftFree(d_scratch);

}

void GPUPlanManager::forward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs){
    complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * ng * nFFTs);
    complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * ng * nFFTs);
    gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * ng * nFFTs,gpuMemcpyHostToDevice);
    forward(d_data,d_scratch,ng,nFFTs);
    gpuMemcpy(scratch,d_scratch,sizeof(complexFloatDevice) * ng * nFFTs,gpuMemcpyDeviceToHost);
    swfftFree(d_data);
    swfftFree(d_scratch);
}

void GPUPlanManager::backward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs){
    if (gpufftExecZ2Z(find_plan(ng,nFFTs,GPUFFT_Z2Z),data,scratch,GPUFFT_INVERSE) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z BACKWARD failed\n");
        return;
    }
}

void GPUPlanManager::backward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs){
    if (gpufftExecC2C(find_plan(ng,nFFTs,GPUFFT_C2C),data,scratch,GPUFFT_INVERSE) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C BACKWARD failed\n");
        return;
    }
}

void GPUPlanManager::backward(complexDoubleHost* data, complexDoubleHost* scratch, int ng, int nFFTs){
    complexDoubleDevice* d_data; swfftAlloc(&d_data,sizeof(complexDoubleDevice) * ng * nFFTs);
    complexDoubleDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexDoubleDevice) * ng * nFFTs);
    gpuMemcpy(d_data,data,sizeof(complexDoubleDevice) * ng * nFFTs,gpuMemcpyHostToDevice);
    backward(d_data,d_scratch,ng,nFFTs);
    gpuMemcpy(scratch,d_scratch,sizeof(complexDoubleDevice) * ng * nFFTs,gpuMemcpyDeviceToHost);
    swfftFree(d_data);
    swfftFree(d_scratch);

}

void GPUPlanManager::backward(complexFloatHost* data, complexFloatHost* scratch, int ng, int nFFTs){
    complexFloatDevice* d_data; swfftAlloc(&d_data,sizeof(complexFloatDevice) * ng * nFFTs);
    complexFloatDevice* d_scratch; swfftAlloc(&d_scratch,sizeof(complexFloatDevice) * ng * nFFTs);
    gpuMemcpy(d_data,data,sizeof(complexFloatDevice) * ng * nFFTs,gpuMemcpyHostToDevice);
    backward(d_data,d_scratch,ng,nFFTs);
    gpuMemcpy(scratch,d_scratch,sizeof(complexFloatDevice) * ng * nFFTs,gpuMemcpyDeviceToHost);
    swfftFree(d_data);
    swfftFree(d_scratch);
}

#endif