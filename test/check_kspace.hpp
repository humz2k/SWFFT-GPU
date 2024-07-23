#include "swfft.hpp"
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <string.h>

using namespace SWFFT;

void assign_delta(complexDoubleHost* data, size_t buff_sz) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    for (size_t i = 0; i < buff_sz; i++) {
        data[i].x = 0;
        data[i].y = 0;
    }
    if (world_rank == 0) {
        data[0].x = 1;
        data[0].y = 0;
    }
}

void assign_delta(complexFloatHost* data, size_t buff_sz) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    for (size_t i = 0; i < buff_sz; i++) {
        data[i].x = 0;
        data[i].y = 0;
    }
    if (world_rank == 0) {
        data[0].x = 1;
        data[0].y = 0;
    }
}

#ifdef SWFFT_GPU
void assign_delta(complexDoubleDevice* data, size_t buff_sz) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    gpuMemset(data, 0, sizeof(complexDoubleDevice) * buff_sz);
    if (world_rank == 0) {
        complexDoubleDevice start;
        start.x = 1;
        start.y = 0;
        gpuMemcpy(data, &start, sizeof(complexDoubleDevice),
                  cudaMemcpyHostToDevice);
    }
}

void assign_delta(complexFloatDevice* data, size_t buff_sz) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    gpuMemset(data, 0, sizeof(complexFloatDevice) * buff_sz);
    if (world_rank == 0) {
        complexFloatDevice start;
        start.x = 1;
        start.y = 0;
        gpuMemcpy(data, &start, sizeof(complexFloatDevice),
                  cudaMemcpyHostToDevice);
    }
}
#endif

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

void allreduce(double* local, double* global, int n, MPI_Op op, MPI_Comm comm) {
    MPI_Allreduce(local, global, 1, MPI_DOUBLE, op, comm);
}

void allreduce(float* local, float* global, int n, MPI_Op op, MPI_Comm comm) {
    MPI_Allreduce(local, global, 1, MPI_FLOAT, op, comm);
}

template <class FFT, class T> bool check_kspace_(FFT& fft, T* a) {

    T LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
    LocalRealMin = LocalRealMax = a[2];
    LocalImagMin = LocalImagMax = a[3];

    for (size_t local_indx = 0; local_indx < fft.buff_sz(); local_indx++) {
        T re = a[local_indx * 2];
        T im = a[local_indx * 2 + 1];

        LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
        LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
        LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
        LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
    }

    const MPI_Comm comm = fft.comm();

    T GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
    allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_MIN, comm);
    allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_MIN, comm);
    allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_MIN, comm);
    allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_MIN, comm);

#ifndef CHECK_SILENT
    if (fft.rank() == 0) {
        std::cout << "k-space:" << std::endl
                  << "      real in " << std::scientific << "[" << GlobalRealMin
                  << "," << GlobalRealMax << "]"
                  << " = " << std::hex << "[" << f2u(GlobalRealMin) << ","
                  << f2u(GlobalRealMax) << "]" << std::endl
                  << "      imag in " << std::scientific << "[" << GlobalImagMin
                  << "," << GlobalImagMax << "]"
                  << " = " << std::hex << "[" << f2u(GlobalImagMin) << ","
                  << f2u(GlobalImagMax) << "]" << std::endl
                  << "   " << std::fixed;
    }
#endif

    if ((round(GlobalRealMin) == 1) && (round(GlobalRealMax) == 1) &&
        (round(GlobalImagMin) == 0) && (round(GlobalImagMax) == 0))
        return true;
    return false;
}

template <class FFT, class T> bool check_rspace_(FFT& fft, T* a) {

    T LocalRealMin, LocalRealMax, LocalImagMin, LocalImagMax;
    LocalRealMin = LocalRealMax = a[2];
    LocalImagMin = LocalImagMax = a[3];

    int start = 0;
    if (fft.rank() == 0) {
        start = 1;
    }

    for (size_t local_indx = start; local_indx < fft.buff_sz(); local_indx++) {
        T re = a[local_indx * 2];
        T im = a[local_indx * 2 + 1];

        LocalRealMin = re < LocalRealMin ? re : LocalRealMin;
        LocalRealMax = re > LocalRealMax ? re : LocalRealMax;
        LocalImagMin = im < LocalImagMin ? im : LocalImagMin;
        LocalImagMax = im > LocalImagMax ? im : LocalImagMax;
    }

    const MPI_Comm comm = fft.comm();

    T GlobalRealMin, GlobalRealMax, GlobalImagMin, GlobalImagMax;
    allreduce(&LocalRealMin, &GlobalRealMin, 1, MPI_MIN, comm);
    allreduce(&LocalRealMax, &GlobalRealMax, 1, MPI_MIN, comm);
    allreduce(&LocalImagMin, &GlobalImagMin, 1, MPI_MIN, comm);
    allreduce(&LocalImagMax, &GlobalImagMax, 1, MPI_MIN, comm);

#ifndef CHECK_SILENT
    if (fft.rank() == 0) {
        std::cout << "r-space:" << std::endl
                  << "      a[0,0,0] = (" << std::fixed << a[0] << "," << a[1]
                  << ")" << std::hex << " = (" << f2u(a[0]) << "," << f2u(a[1])
                  << ")" << std::endl
                  << "      real in " << std::scientific << "[" << GlobalRealMin
                  << "," << GlobalRealMax << "]"
                  << " = " << std::hex << "[" << f2u(GlobalRealMin) << ","
                  << f2u(GlobalRealMax) << "]" << std::endl
                  << "      imag in " << std::scientific << "[" << GlobalImagMin
                  << "," << GlobalImagMax << "]"
                  << " = " << std::hex << "[" << f2u(GlobalImagMin) << ","
                  << f2u(GlobalImagMax) << "]" << std::endl
                  << "   " << std::fixed;
    }
#endif

    bool base_correct =
        ((round(GlobalRealMin) == 0) && (round(GlobalRealMax) == 0) &&
         (round(GlobalImagMin) == 0) && (round(GlobalImagMax) == 0));
    if (fft.rank() == 0) {
        bool center_correct =
            (round(a[0]) == fft.ngx() * fft.ngy() * fft.ngz()) &&
            (round(a[1]) == 0);
        base_correct = base_correct && center_correct;
    }
    return base_correct;
}

#ifdef SWFFT_GPU
template <class FFT> bool check_kspace(FFT& fft, complexDoubleDevice* a_) {
    double* a = (double*)malloc(sizeof(complexDoubleDevice) * fft.buff_sz());
    gpuMemcpy(a, a_, sizeof(complexDoubleDevice) * fft.buff_sz(),
              gpuMemcpyDeviceToHost);
    bool out = check_kspace_(fft, a);
    free(a);
    return out;
}

template <class FFT> bool check_kspace(FFT& fft, complexFloatDevice* a_) {
    float* a = (float*)malloc(sizeof(complexFloatDevice) * fft.buff_sz());
    gpuMemcpy(a, a_, sizeof(complexFloatDevice) * fft.buff_sz(),
              gpuMemcpyDeviceToHost);
    bool out = check_kspace_(fft, a);
    free(a);
    return out;
}
#endif

template <class FFT> bool check_kspace(FFT& fft, complexDoubleHost* a) {
    return check_kspace_(fft, (double*)a);
}

template <class FFT> bool check_kspace(FFT& fft, complexFloatHost* a) {
    return check_kspace_(fft, (float*)a);
}

#ifdef SWFFT_GPU
template <class FFT> bool check_rspace(FFT& fft, complexDoubleDevice* a_) {
    double* a = (double*)malloc(sizeof(complexDoubleDevice) * fft.buff_sz());
    gpuMemcpy(a, a_, sizeof(complexDoubleDevice) * fft.buff_sz(),
              gpuMemcpyDeviceToHost);
    bool out = check_rspace_(fft, a);
    free(a);
    return out;
}

template <class FFT> bool check_rspace(FFT& fft, complexFloatDevice* a_) {
    float* a = (float*)malloc(sizeof(complexFloatDevice) * fft.buff_sz());
    gpuMemcpy(a, a_, sizeof(complexFloatDevice) * fft.buff_sz(),
              gpuMemcpyDeviceToHost);
    bool out = check_rspace_(fft, a);
    free(a);
    return out;
}
#endif

template <class FFT> bool check_rspace(FFT& fft, complexDoubleHost* a) {
    return check_rspace_(fft, (double*)a);
}

template <class FFT> bool check_rspace(FFT& fft, complexFloatHost* a) {
    return check_rspace_(fft, (float*)a);
}