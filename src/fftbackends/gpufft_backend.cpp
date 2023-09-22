#ifdef SWFFT_FFTW
#undef SWFFT_FFTW
#endif
#include "fftwrangler.hpp"

#include <mpi.h>
#ifdef SWFFT_CUFFT
#ifdef SWFFT_GPU
#include <stdio.h>
#include <stdlib.h>
namespace SWFFT{
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

GPUPlanManager::GPUPlanManager(){
    for (int i = 0; i < N_FFT_CACHE; i++){
        plans[i].valid = false;
        plans[i].plan = 0;
    }
}

GPUPlanManager::~GPUPlanManager(){
    for (int i = 0; i < N_FFT_CACHE; i++){
        if (plans[i].valid){
            if (gpufftDestroy(plans[i].plan) != GPUFFT_SUCCESS){
                printf("CUFFT error: Couldn't destory plan!\n");
            }
        }
    }
}

void GPUPlanManager::query(){
    printf("Using gpuFFT\n");
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
                printf("CUFFT error: Plan creation failed with (ng = %d, nFFTs = %d)\n",ng,nFFTs);
            }
            return plans[i].plan;

        }
    }
    printf("Out of space for plans!\n");
    //MPI_Abort(MPI_COMM_WORLD,1);
    return NULL;
}

void GPUPlanManager::forward(complexDoubleDevice* data, complexDoubleDevice* scratch, int ng, int nFFTs){
    if (gpufftExecZ2Z(find_plan(ng,nFFTs,GPUFFT_Z2Z),data,scratch,GPUFFT_FORWARD) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecZ2Z FORWARD failed\n");
        return;
    }
    //gpuDeviceSynchronize();
}

void GPUPlanManager::forward(complexFloatDevice* data, complexFloatDevice* scratch, int ng, int nFFTs){
    if (gpufftExecC2C(find_plan(ng,nFFTs,GPUFFT_C2C),data,scratch,GPUFFT_FORWARD) != GPUFFT_SUCCESS){
        printf("CUFFT error: ExecC2C FORWARD failed\n");
        return;
    }
    //gpuDeviceSynchronize();
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
}
#endif
#endif