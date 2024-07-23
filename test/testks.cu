#include "check_kspace.hpp"
#include "swfft.hpp"
#include <stdio.h>
#include <stdlib.h>

using namespace SWFFT;

static int n_tests = 0;
static int n_passed = 0;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define BLOCKSIZE 64

template<class T, class SWFFT_T>
void fill_test_cpu(T* data, SWFFT_T& my_swfft){
    auto dist3d = my_swfft.dist3d();
    for (size_t i = 0; i < my_swfft.buff_sz(); i++) {
        int3 rs = dist3d.get_rs(i);
        int tmp = rs.x * my_swfft.ngy() * my_swfft.ngz() + rs.y * my_swfft.ngz() + rs.z;
        data[i].x = tmp;
        data[i].y = 0;
    }
}

template<class T, class d3d>
__global__ void fill_test_gpu_kernel(T* data, d3d dist3d, int ngy, int ngz, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    int3 rs = dist3d.get_rs(i);
    int tmp = rs.x * ngy * ngz + rs.y * ngz + rs.z;
    data[i].x = tmp;
    data[i].y = 0;
}

template<class T, class SWFFT_T>
void fill_test_gpu(T* data, SWFFT_T& my_swfft){
    int blockSize = 32;
    int numBlocks = ((int)my_swfft.buff_sz() + (blockSize - 1)) / blockSize;
    gpuLaunch(fill_test_gpu_kernel,numBlocks,blockSize,data,my_swfft.dist3d(),my_swfft.ngy(),my_swfft.ngz(),my_swfft.buff_sz());
    gpuDeviceSynchronize();
}

template<class SWFFT_T>
void fill_test(complexDoubleDevice* data, SWFFT_T& my_swfft){
    fill_test_gpu(data,my_swfft);
}

template<class SWFFT_T>
void fill_test(complexFloatDevice* data, SWFFT_T& my_swfft){
    fill_test_gpu(data,my_swfft);
}

template<class SWFFT_T>
void fill_test(complexDoubleHost* data, SWFFT_T& my_swfft){
    fill_test_cpu(data,my_swfft);
}

template<class SWFFT_T>
void fill_test(complexFloatHost* data, SWFFT_T& my_swfft){
    fill_test_cpu(data,my_swfft);
}

template<class T, class SWFFT_T>
bool check_ks_cpu(T* data, SWFFT_T& my_swfft){
    auto dist3d = my_swfft.dist3d();
    int local = 1;
    int global;
    for (size_t i = 0; i < my_swfft.buff_sz(); i++) {
        int3 ks = dist3d.get_ks(i);
        int tmp = ks.x * my_swfft.ngy() * my_swfft.ngz() + ks.y * my_swfft.ngz() + ks.z;
        if (data[i].x != (double)tmp) {
            local = 0;
            break;
        }
    }
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, my_swfft.comm());
    bool passed = global == my_swfft.world_size();

    if (my_swfft.rank() == 0) {
        if (passed) {
            printf("Passed ks\n   ");
        } else {
            printf("Failed ks...\n   ");
        }
    }

    return passed;
}

template<class T, class d3d>
__global__ void check_ks_gpu_kernel(T* data, d3d dist3d, int ngy, int ngz, int n, int* out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    int3 ks = dist3d.get_ks(i);
    int tmp = ks.x * ngy * ngz + ks.y * ngz + ks.z;
    if (data[i].x != (double)tmp){
        atomicAdd(out,1);
    }
}

template<class T, class SWFFT_T>
bool check_ks_gpu(T* data, SWFFT_T& my_swfft){
    int blockSize = 32;
    int numBlocks = ((int)my_swfft.buff_sz() + (blockSize - 1)) / blockSize;
    auto dist3d = my_swfft.dist3d();
    int local = 1;
    int global;
    int* d_local;
    gpuMalloc(&d_local,sizeof(int));
    gpuMemset(d_local,0,sizeof(int));
    gpuLaunch(check_ks_gpu_kernel,numBlocks,blockSize,data,my_swfft.dist3d(),my_swfft.ngy(),my_swfft.ngz(),my_swfft.buff_sz(),d_local);
    gpuDeviceSynchronize();
    gpuMemcpy(&local,d_local,sizeof(int),gpuMemcpyDeviceToHost);
    gpuFree(d_local);
    local = local == 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, my_swfft.comm());
    bool passed = global == my_swfft.world_size();
    if (my_swfft.rank() == 0) {
        if (passed) {
            printf("Passed ks\n   ");
        } else {
            printf("Failed ks...\n   ");
        }
    }

    return passed;
}

template<class SWFFT_T>
bool check_ks(complexDoubleDevice* data, SWFFT_T& my_swfft){
    return check_ks_gpu(data,my_swfft);
}

template<class SWFFT_T>
bool check_ks(complexFloatDevice* data, SWFFT_T& my_swfft){
    return check_ks_gpu(data,my_swfft);
}

template<class SWFFT_T>
bool check_ks(complexDoubleHost* data, SWFFT_T& my_swfft){
    return check_ks_cpu(data,my_swfft);
}

template<class SWFFT_T>
bool check_ks(complexFloatHost* data, SWFFT_T& my_swfft){
    return check_ks_cpu(data,my_swfft);
}

template<class T, class SWFFT_T>
bool check_rs_cpu(T* data, SWFFT_T& my_swfft){
    auto dist3d = my_swfft.dist3d();
    int local = 1;
    int global;
    for (size_t i = 0; i < my_swfft.buff_sz(); i++) {
        int3 rs = dist3d.get_rs(i);
        int tmp = rs.x * my_swfft.ngy() * my_swfft.ngz() + rs.y * my_swfft.ngz() + rs.z;
        if (data[i].x != (double)tmp) {
            local = 0;
            break;
        }
    }
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, my_swfft.comm());
    bool passed = global == my_swfft.world_size();

    if (my_swfft.rank() == 0) {
        if (passed) {
            printf("Passed rs\n   ");
        } else {
            printf("Failed rs...\n   ");
        }
    }

    return passed;
}

template<class T, class d3d>
__global__ void check_rs_gpu_kernel(T* data, d3d dist3d, int ngy, int ngz, int n, int* out){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n)
        return;
    int3 rs = dist3d.get_rs(i);
    int tmp = rs.x * ngy * ngz + rs.y * ngz + rs.z;
    if (data[i].x != (double)tmp){
        atomicAdd(out,1);
    }
}

template<class T, class SWFFT_T>
bool check_rs_gpu(T* data, SWFFT_T& my_swfft){
    int blockSize = 32;
    int numBlocks = ((int)my_swfft.buff_sz() + (blockSize - 1)) / blockSize;
    auto dist3d = my_swfft.dist3d();
    int local = 1;
    int global;
    int* d_local;
    gpuMalloc(&d_local,sizeof(int));
    gpuMemset(d_local,0,sizeof(int));
    gpuLaunch(check_rs_gpu_kernel,numBlocks,blockSize,data,my_swfft.dist3d(),my_swfft.ngy(),my_swfft.ngz(),my_swfft.buff_sz(),d_local);
    gpuDeviceSynchronize();
    gpuMemcpy(&local,d_local,sizeof(int),gpuMemcpyDeviceToHost);
    gpuFree(d_local);
    local = local == 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_SUM, my_swfft.comm());
    bool passed = global == my_swfft.world_size();
    if (my_swfft.rank() == 0) {
        if (passed) {
            printf("Passed rs\n   ");
        } else {
            printf("Failed rs...\n   ");
        }
    }

    return passed;
}

template<class SWFFT_T>
bool check_rs(complexDoubleDevice* data, SWFFT_T& my_swfft){
    return check_rs_gpu(data,my_swfft);
}

template<class SWFFT_T>
bool check_rs(complexFloatDevice* data, SWFFT_T& my_swfft){
    return check_rs_gpu(data,my_swfft);
}

template<class SWFFT_T>
bool check_rs(complexDoubleHost* data, SWFFT_T& my_swfft){
    return check_rs_cpu(data,my_swfft);
}

template<class SWFFT_T>
bool check_rs(complexFloatHost* data, SWFFT_T& my_swfft){
    return check_rs_cpu(data,my_swfft);
}

template <class SWFFT_T, class T>
bool test(bool k_in_blocks, int ngx, int ngy = 0, int ngz = 0) {
    ngy = (ngy == 0) ? ngx : ngy;
    ngz = (ngz == 0) ? ngx : ngz;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    n_tests++;
    if (world_rank == 0)
        printf(
            "Testing %s with T = %s, k_in_blocks = %d and ng = [%d %d %d]\n   ",
            typeid(SWFFT_T).name(), typeid(T).name(), k_in_blocks, ngx, ngy,
            ngz);

    SWFFT_T my_swfft(MPI_COMM_WORLD, ngx, ngy, ngz, BLOCKSIZE, k_in_blocks);

    T* data;
    swfftAlloc(&data, sizeof(T) * my_swfft.buff_sz());
    T* scratch;
    swfftAlloc(&scratch, sizeof(T) * my_swfft.buff_sz());

    fill_test(data,my_swfft);

    my_swfft.forward(data, scratch);

    bool ks_test = check_ks(data,my_swfft);

    my_swfft.backward(data, scratch);

    bool rs_test = check_rs(data,my_swfft);

    if (!world_rank){
        if (ks_test && rs_test) {
            printf("Passed!\n\n");
            n_passed++;
        } else
            printf("Failed...\n\n");
    }

    swfftFree(data);
    swfftFree(scratch);

    return ks_test && rs_test;
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
#ifdef SWFFT_GPU
    gpuFree(0);
#endif

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (!((argc == 2) || (argc == 4))) {
        if (world_rank == 0)
            printf("USAGE: %s <ngx> [ngy ngz]\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    int ngx = atoi(argv[1]);
    int ngy = ngx;
    int ngz = ngx;
    if (argc == 4) {
        ngy = atoi(argv[2]);
        ngz = atoi(argv[3]);
    }

#ifdef SWFFT_ALLTOALL
#ifdef SWFFT_GPU
    //test<swfft<AllToAllGPU, CPUMPI, TestFFT>, complexDoubleHost>(false, ngx,
    //                                                             ngy, ngz);
    test<swfft<AllToAllGPU, CPUMPI, TestFFT>, complexDoubleHost>(true, ngx, ngy,
                                                                 ngz);
    //test<swfft<AllToAllGPU, CPUMPI, TestFFT>, complexDoubleDevice>(false, ngx,
    //                                                             ngy, ngz);
    test<swfft<AllToAllGPU, CPUMPI, TestFFT>, complexDoubleDevice>(true, ngx, ngy,
                                                                 ngz);
#endif
    //test<swfft<AllToAllCPU, CPUMPI, TestFFT>, complexDoubleHost>(false, ngx,
    //                                                             ngy, ngz);
    test<swfft<AllToAllCPU, CPUMPI, TestFFT>, complexDoubleHost>(true, ngx, ngy,
                                                                 ngz);
#endif
#ifdef SWFFT_HQFFT
#ifdef SWFFT_GPU
    test<swfft<HQA2AGPU, CPUMPI, TestFFT>, complexDoubleHost>(false, ngx, ngy,
                                                              ngz);
    test<swfft<HQA2AGPU, CPUMPI, TestFFT>, complexDoubleHost>(true, ngx, ngy,
                                                              ngz);
    test<swfft<HQPairGPU, CPUMPI, TestFFT>, complexDoubleHost>(false, ngx, ngy,
                                                               ngz);
    test<swfft<HQPairGPU, CPUMPI, TestFFT>, complexDoubleHost>(true, ngx, ngy,
                                                               ngz);
    test<swfft<HQA2AGPU, CPUMPI, TestFFT>, complexDoubleDevice>(false, ngx, ngy,
                                                              ngz);
    test<swfft<HQA2AGPU, CPUMPI, TestFFT>, complexDoubleDevice>(true, ngx, ngy,
                                                              ngz);
    test<swfft<HQPairGPU, CPUMPI, TestFFT>, complexDoubleDevice>(false, ngx, ngy,
                                                               ngz);
    test<swfft<HQPairGPU, CPUMPI, TestFFT>, complexDoubleDevice>(true, ngx, ngy,
                                                               ngz);
#endif
#endif
#ifdef SWFFT_PAIRWISE
    test<swfft<Pairwise, CPUMPI, TestFFT>, complexDoubleHost>(false, ngx, ngy,
                                                              ngz);
    test<swfft<Pairwise, CPUMPI, TestFFT>, complexDoubleDevice>(false, ngx, ngy,
                                                              ngz);
#endif

    if (world_rank == 0)
        printf("%d/%d tests passed\n", n_passed, n_tests);
    MPI_Finalize();
    return 0;
}