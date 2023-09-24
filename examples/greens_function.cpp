#include "swfft.hpp"
#include <stdio.h>
#include <stdlib.h>

using namespace SWFFT; //for convenience

///
// Cache 2nd-order discrete Green's function
//
// G(k) = 1 / (2 * (Sum_i cos(2 pi k_i / n) - 3))
///
void cache_greens(swfft<Pairwise,CPUMPI,fftw>& fft, double* greens_function){
    
    
    double coeff = 0.5 / ((double)fft.global_size()); //prefactor

    double tpi = 2.0*atan(1.0)*4.0; 

    double kstep[3];
    kstep[0] = tpi / ((double)fft.ngx()); //2pi / n
    kstep[1] = tpi / ((double)fft.ngy()); //2pi / n
    kstep[2] = tpi / ((double)fft.ngz()); //2pi / n

    for (int i = 0; i < fft.buff_sz(); i++){
        int3 ks = fft.get_ks(i); //k_i
        double green;
        //handle the pole
        //this should really be done outside of the loop
        if ((ks.x == 0) && (ks.y == 0) && (ks.z == 0)){
            green = 0;
        } else {
            //calculate green
            green = coeff / (cos(ks.x * kstep[0]) + cos(ks.y * kstep[1]) + cos(ks.z * kstep[2]) - 3.0);
        }
        greens_function[i] = green; //cache in greens_function
    }

}
///
// Solve for potential by applying the Green's functino to the
// density (in k-space)
///
void kspace_solve(swfft<Pairwise,CPUMPI,fftw>& fft, double* greens_function, complexDoubleHost* data){

    for (int i = 0; i < fft.buff_sz(); i++){
        
        double green = greens_function[i]; //load green

        complexDoubleHost rho = data[i]; //load rho

        //apply greens function
        complexDoubleHost phi;
        phi.x = rho.x * green;
        phi.y = rho.y * green;

        data[i] = phi; //store phi

    }

}

///
// Fill with random dummy data
///
void fill_random(swfft<Pairwise,CPUMPI,fftw>& fft, complexDoubleHost* data){
    for (int i = 0; i < fft.buff_sz(); i++){
        data[i].x = (double)rand() / (double)RAND_MAX;
        data[i].y = 0;
    }
}

///
// Run a test
///
void run_test(){
    //Initialize swfft
    swfft<Pairwise,CPUMPI,fftw> fft(MPI_COMM_WORLD,256);
    
    //Allocate fft buffers
    complexDoubleHost* data; swfftAlloc(&data,sizeof(complexDoubleHost) * fft.buff_sz());
    complexDoubleHost* scratch; swfftAlloc(&scratch,sizeof(complexDoubleHost) * fft.buff_sz());
    
    //allocate greens function cache
    double* greens_function = (double*)malloc(sizeof(double) * fft.buff_sz());

    //cache greens function
    cache_greens(fft,greens_function);

    //fill with dummy data
    fill_random(fft,data);

    //do a forward fft on the data
    fft.forward(data,scratch);

    //solve for phi
    kspace_solve(fft,greens_function,data);

    //do a backward fft
    fft.backward(data,scratch);

}

int main(){
    MPI_Init(NULL,NULL);

    run_test();

    MPI_Finalize();
}