#include <mpi.h>
#include "swfft.hpp"
#include <string.h>
#include <iostream>
#include <math.h>

uint64_t f2u(double d) {
  uint64_t i;
  memcpy(&i, &d, 8);
  return i;
}

uint32_t f2u(float d) {
  uint32_t i;
  memcpy(&i, &d, 4);
  return i;
}

void allreduce(double* local, double* global, int n, MPI_Op op, MPI_Comm comm){
    MPI_Allreduce(local, global, 1, MPI_DOUBLE, op, comm);
}

void allreduce(float* local, float* global, int n, MPI_Op op, MPI_Comm comm){
    MPI_Allreduce(local, global, 1, MPI_FLOAT, op, comm);
}

template<class FFT, class T>
bool check_kspace_(FFT &fft, T *a){

    T LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
    LocalRealMin = LocalRealMax = a[2];
    LocalImagMin = LocalImagMax = a[3];

    for(int local_indx=0; local_indx<fft.buff_sz(); local_indx++) {
        T re = a[local_indx*2];
        T im = a[local_indx*2 + 1];

        LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
        LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
        LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
        LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
    }

    const MPI_Comm comm = fft.comm();

    T GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
    allreduce(&LocalRealMin,&GlobalRealMin,1,MPI_MIN,comm);
    allreduce(&LocalRealMax,&GlobalRealMax,1,MPI_MIN,comm);
    allreduce(&LocalImagMin,&GlobalImagMin,1,MPI_MIN,comm);
    allreduce(&LocalImagMax,&GlobalImagMax,1,MPI_MIN,comm);

    if(fft.rank() == 0) {
    std::cout << "k-space:" << std::endl
            << "      real in " << std::scientific
            << "[" << GlobalRealMin << "," << GlobalRealMax << "]"
            << " = " << std::hex
            << "[" << f2u(GlobalRealMin) << ","
            << f2u(GlobalRealMax) << "]"
            << std::endl
            << "      imag in " << std::scientific
            << "[" << GlobalImagMin << "," << GlobalImagMax << "]"
            << " = " << std::hex
                << "[" << f2u(GlobalImagMin) << ","
            << f2u(GlobalImagMax) << "]"
            << std::endl << "   " << std::fixed;
    }

    if ((round(GlobalRealMin) == 1) && (round(GlobalRealMax) == 1) && (round(GlobalImagMin) == 0) && (round(GlobalImagMax) == 0))return true;
    return false;

}

#ifdef GPU
template<class FFT>
bool check_kspace(FFT& fft, complexDoubleDevice* a_){
    double* a = (double*)malloc(sizeof(complexDoubleDevice) * fft.buff_sz());
    gpuMemcpy(a,a_,sizeof(complexDoubleDevice)*fft.buff_sz(),gpuMemcpyDeviceToHost);
    bool out = check_kspace_(fft,a);
    free(a);
    return out;
}

template<class FFT>
bool check_kspace(FFT& fft, complexFloatDevice* a_){
    float* a = (float*)malloc(sizeof(complexFloatDevice) * fft.buff_sz());
    gpuMemcpy(a,a_,sizeof(complexFloatDevice)*fft.buff_sz(),gpuMemcpyDeviceToHost);
    bool out = check_kspace_(fft,a);
    free(a);
    return out;
}
#endif

template<class FFT>
bool check_kspace(FFT& fft, complexDoubleHost* a){
    return check_kspace_(fft,(double*)a);
}

template<class FFT>
bool check_kspace(FFT& fft, complexFloatHost* a){
    return check_kspace_(fft,(float*)a);
}