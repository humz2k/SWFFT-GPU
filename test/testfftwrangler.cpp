#include "fftwrangler.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define IS_TRUE(func,T,PlanManager,NG) { std::cout << TOSTRING(func) << "<" << TOSTRING(T) << "," << TOSTRING(PlanManager) << ">(" << TOSTRING(NG) << ")"; if (!(func<T,PlanManager>(NG))){ std::cout << " failed on line " << __LINE__ << std::endl;}else{std::cout  << " passed " << std::endl;} }

template<class T, class PlanManager>
bool test_fftwrangler(int NG){

    PlanManager plan_manager;

    double real = 0;
    double complex = 0;

    T* data; swfftAlloc(&data,NG*sizeof(T));
    T* scratch; swfftAlloc(&scratch,NG*sizeof(T));

    for (int i = 1; i < NG; i++){
        data[i].x = 0;
        data[i].y = 0;
    }
    data[0].x = 1;
    data[0].y = 0;

    plan_manager.forward(data,scratch,NG,1);
    if (((int)real != 0) || ((int)complex != 0)){
        swfftFree(data);
        swfftFree(scratch);
        return false;
    }

    plan_manager.backward(scratch,data,NG,1);

    real = 0;
    complex = 0;
    for (int i = 1; i < NG; i++){
        real += data[i].x;
        complex += data[i].y;
    }

    if (((int)real != 0) || ((int)complex != 0) || ((int)data[0].x != NG) || ((int)data[0].y != 0)){
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
    return 0;
}