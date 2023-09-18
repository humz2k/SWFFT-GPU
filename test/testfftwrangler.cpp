#include "fftwrangler.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define IS_TRUE(func,T,PlanManager,NG) { std::cout << TOSTRING(func) << "<" << TOSTRING(T) << "," << TOSTRING(PlanManager) << ">(" << TOSTRING(NG) << ")"; if (!(func<T,PlanManager>(NG))){ std::cout << " failed on line " << __LINE__ << std::endl;}else{std::cout  << " passed " << std::endl;} }

void assign_delta(complexDoubleHost* data, int NG){
    for (int i = 1; i < NG; i++){
        data[i].x = 0;
        data[i].y = 0;
    }
    data[0].x = 1;
    data[0].y = 0;
}

void assign_delta(complexFloatHost* data, int NG){
    for (int i = 1; i < NG; i++){
        data[i].x = 0;
        data[i].y = 0;
    }
    data[0].x = 1;
    data[0].y = 0;
}

void assign_delta(complexDoubleDevice* data, int NG){
    gpuMemset(data,0,sizeof(complexDoubleDevice)*NG);
    complexDoubleDevice start;
    start.x = 1;
    start.y = 0;
    gpuMemcpy(data,&start,sizeof(complexDoubleDevice),cudaMemcpyHostToDevice);
}

void assign_delta(complexFloatDevice* data, int NG){
    gpuMemset(data,0,sizeof(complexFloatDevice)*NG);
    complexFloatDevice start;
    start.x = 1;
    start.y = 0;
    gpuMemcpy(data,&start,sizeof(complexFloatDevice),cudaMemcpyHostToDevice);
}

bool check_k(complexDoubleHost* data, int NG){
    for (int i = 1; i < NG; i++){
        if ((((int)data[i].x) != 1) || (((int)data[i].y) != 0)){
            return false;
        }
    }
    return true;
}

bool check_k(complexFloatHost* data, int NG){
    for (int i = 1; i < NG; i++){
        if ((((int)data[i].x) != 1) || (((int)data[i].y) != 0)){
            return false;
        }
    }
    return true;
}

bool check_k(complexDoubleDevice* d_data, int NG){
    complexDoubleHost* h_data; swfftAlloc(&h_data, sizeof(complexDoubleHost) * NG);
    gpuMemcpy(h_data,d_data,sizeof(complexDoubleDevice)*NG,gpuMemcpyDeviceToHost);
    bool out = check_k(h_data,NG);
    swfftFree(h_data);
    return out;
}

bool check_k(complexFloatDevice* d_data, int NG){
    complexFloatHost* h_data; swfftAlloc(&h_data, sizeof(complexFloatHost) * NG);
    gpuMemcpy(h_data,d_data,sizeof(complexFloatDevice)*NG,gpuMemcpyDeviceToHost);
    bool out = check_k(h_data,NG);
    swfftFree(h_data);
    return out;
}

bool check_r(complexDoubleHost* data, int NG){
    double real = 0;
    double complex = 0;

    for (int i = 1; i < NG; i++){
        real += data[i].x;
        complex += data[i].y;
    }

    if (((int)real != 0) || ((int)complex != 0) || ((int)data[0].x != NG) || ((int)data[0].y != 0)){
        return false;
    }

    return true;
}

bool check_r(complexFloatHost* data, int NG){
    double real = 0;
    double complex = 0;

    for (int i = 1; i < NG; i++){
        real += data[i].x;
        complex += data[i].y;
    }

    if (((int)real != 0) || ((int)complex != 0) || ((int)data[0].x != NG) || ((int)data[0].y != 0)){
        return false;
    }

    return true;
}

bool check_r(complexDoubleDevice* d_data, int NG){
    complexDoubleHost* h_data; swfftAlloc(&h_data, sizeof(complexDoubleHost) * NG);
    gpuMemcpy(h_data,d_data,sizeof(complexDoubleDevice)*NG,gpuMemcpyDeviceToHost);
    bool out = check_r(h_data,NG);
    swfftFree(h_data);
    return out;
}

bool check_r(complexFloatDevice* d_data, int NG){
    complexFloatHost* h_data; swfftAlloc(&h_data, sizeof(complexFloatHost) * NG);
    gpuMemcpy(h_data,d_data,sizeof(complexFloatDevice)*NG,gpuMemcpyDeviceToHost);
    bool out = check_r(h_data,NG);
    swfftFree(h_data);
    return out;
}

template<class T, class PlanManager>
bool test_fftwrangler(int NG){

    PlanManager plan_manager;

    double real = 0;
    double complex = 0;

    T* data; swfftAlloc(&data,NG*sizeof(T));
    T* scratch; swfftAlloc(&scratch,NG*sizeof(T));

    assign_delta(data,NG);

    plan_manager.forward(data,scratch,NG,1);

    if (!check_k(scratch,NG)){
        swfftFree(data);
        swfftFree(scratch);
        return false;
    }

    plan_manager.backward(scratch,data,NG,1);

    if (!check_r(data,NG)){
        swfftFree(data);
        swfftFree(scratch);
        return false;
    }

    swfftFree(data);
    swfftFree(scratch);
    return true;
}

int main(){
    IS_TRUE(test_fftwrangler,complexDoubleHost,FFTWPlanManager,64);
    IS_TRUE(test_fftwrangler,complexFloatHost,FFTWPlanManager,64);
    IS_TRUE(test_fftwrangler,complexDoubleDevice,FFTWPlanManager,64);
    IS_TRUE(test_fftwrangler,complexFloatDevice,FFTWPlanManager,64);
    IS_TRUE(test_fftwrangler,complexDoubleHost,GPUPlanManager,64);
    IS_TRUE(test_fftwrangler,complexFloatHost,GPUPlanManager,64);
    IS_TRUE(test_fftwrangler,complexDoubleDevice,GPUPlanManager,64);
    IS_TRUE(test_fftwrangler,complexFloatDevice,GPUPlanManager,64);
    return 0;
}