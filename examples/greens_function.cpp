#include "swfft.hpp"
#include <stdio.h>
#include <stdlib.h>

using namespace SWFFT;

void cache_greens(swfft<Pairwise,CPUMPI,fftw>& fft, double* greens_function){
    
    double coeff = 0.5 / ((double)fft.global_size());

    double tpi = 2.0*atan(1.0)*4.0;

    double kstep[3];
    kstep[0] = tpi / ((double)fft.ngx());
    kstep[1] = tpi / ((double)fft.ngy());
    kstep[2] = tpi / ((double)fft.ngz());

    for (int i = 0; i < fft.buff_sz(); i++){
        int3 ks = fft.get_ks(i);
        double green = coeff / (cos(ks.x * kstep[0]) + cos(ks.y * kstep[1]) + cos(ks.z * kstep[2]) - 3.0);
        greens_function[i] = green;
    }

}

void kspace_solve(swfft<Pairwise,CPUMPI,fftw>& fft, double* greens_function, complexDoubleHost* data){

    for (int i = 0; i < fft.buff_sz(); i++){
        
        double green = greens_function[i];

        complexDoubleHost rho = data[i];

        rho.x *= green;
        rho.y *= green;

        data[i] = rho;

    }

}

void fill_random(swfft<Pairwise,CPUMPI,fftw>& fft, complexDoubleHost* data){
    for (int i = 0; i < fft.buff_sz(); i++){
        data[i].x = (double)rand() / (double)RAND_MAX;
        data[i].y = 0;
    }
}

void run_test(){
    swfft<Pairwise,CPUMPI,fftw> fft(MPI_COMM_WORLD,256);
    complexDoubleHost* data; swfftAlloc(&data,sizeof(complexDoubleHost) * fft.buff_sz());
    complexDoubleHost* scratch; swfftAlloc(&scratch,sizeof(complexDoubleHost) * fft.buff_sz());
    double* greens_function = (double*)malloc(sizeof(double) * fft.buff_sz());

    cache_greens(fft,greens_function);
    fill_random(fft,data);

    fft.forward(data,scratch);

    kspace_solve(fft,greens_function,data);

    fft.backward(data,scratch);

}

int main(){
    MPI_Init(NULL,NULL);

    run_test();

    MPI_Finalize();
}