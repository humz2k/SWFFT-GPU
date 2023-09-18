#include "fftwrangler.hpp"
#include <stdio.h>
#include <stdlib.h>

#define NG 64

int main(){
    fftw_complex* data = (fftw_complex*)malloc(NG * sizeof(fftw_complex));
    fftw_complex* scratch = (fftw_complex*)malloc(NG * sizeof(fftw_complex));

    for (int i = 1; i < NG; i++){
        data[i][0] = 0;
        data[i][1] = 0;
    }
    data[0][0] = 1;
    data[0][1] = 0;

    FFTWPlanManager plan_manager;
    plan_manager.forward(data,scratch,NG,1);

    double real = 0;
    double complex = 0;
    for (int i = 0; i < NG; i++){
        real += scratch[i][0];
        complex += scratch[i][1];
    }
    printf("real = %g, complex = %g\n",real,complex);

    for (int i = 1; i < NG; i++){
        data[i][0] = 0;
        data[i][1] = 0;
    }
    data[0][0] = 1;
    data[0][1] = 0;

    plan_manager.forward(data,scratch,NG,1);

    real = 0;
    complex = 0;
    for (int i = 0; i < NG; i++){
        real += scratch[i][0];
        complex += scratch[i][1];
    }
    printf("real = %g, complex = %g\n",real,complex);

    plan_manager.backward(scratch,data,NG,1);

    real = 0;
    complex = 0;
    for (int i = 1; i < NG; i++){
        real += data[i][0];
        complex += data[i][1];
    }
    printf("[0] = (%g, %g), real = %g, complex = %g\n",data[0][0],data[0][1],real,complex);

    free(data);
    free(scratch);
}