/**
 * @file hqfft/hqfft.hpp
 */

#ifndef _SWFFT_HQFFT_HPP_
#define _SWFFT_HQFFT_HPP_
#ifdef SWFFT_HQFFT

#include "common/copy_buffers.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "mpi/mpi_isend_irecv.hpp"
#include "mpi/mpiwrangler.hpp"
#include "collectivecomm.hpp"
#include "dfft.hpp"
#include "distribution.hpp"
#include "hqfft_reorder.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>
#include <stdio.h>

namespace SWFFT {
#ifdef SWFFT_GPU
template <class MPI_T, class FFTBackend> class HQA2AGPU : public Backend {
  private:
    HQFFT::Distribution<HQFFT::AllToAll, MPI_T, HQFFT::GPUReshape> dist;
    HQFFT::Dfft<HQFFT::Distribution, HQFFT::GPUReshape, HQFFT::AllToAll, MPI_T,
                FFTBackend>
        dfft;

  public:
    HQA2AGPU() {}

    HQA2AGPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true)
        : dist(comm, ngx, ngx, ngx, blockSize), dfft(dist, ks_as_block) {}

    HQA2AGPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize,
             bool ks_as_block = true)
        : dist(comm, ngx, ngy, ngz, blockSize), dfft(dist, ks_as_block) {}

    ~HQA2AGPU(){};

    void query() {
        printf("Using HQAllToAllGPU\n");
        printf("   distribution = [%d %d %d]\n", dist.dims[0], dist.dims[1],
               dist.dims[2]);
    }

    int3 local_ng() { return dfft.local_ng(); }

    int local_ng(int i) { return dfft.local_ng(i); }

    int3 get_ks(int idx) { return dfft.get_ks(idx); }

    int3 get_rs(int idx) { return dfft.get_rs(idx); }

    int ngx() { return dfft.ng[0]; }

    int ngy() { return dfft.ng[1]; }

    int ngz() { return dfft.ng[2]; }

    int3 ng() { return make_int3(dfft.ng[0], dfft.ng[1], dfft.ng[2]); }

    int ng(int i) { return dfft.ng[i]; }

    size_t buff_sz() { return dist.nlocal; }

    int3 coords() {
        return make_int3(dist.coords[0], dist.coords[1], dist.coords[2]);
    }

    int3 dims() { return make_int3(dist.dims[0], dist.dims[1], dist.dims[2]); }

    int rank() { return dist.world_rank; }

    MPI_Comm comm() { return dist.world_comm; }

    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        dfft.forward(data, scratch);
    }

    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        dfft.forward(data, scratch);
    }

    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        dfft.backward(data, scratch);
    }

    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        dfft.backward(data, scratch);
    }

    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void forward(complexDoubleDevice* data) {
        complexDoubleDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleDevice) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    void forward(complexFloatDevice* data) {
        complexFloatDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatDevice) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    void forward(complexDoubleHost* data) {
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    void forward(complexFloatHost* data) {
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    void backward(complexDoubleDevice* data) {
        complexDoubleDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleDevice) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    void backward(complexFloatDevice* data) {
        complexFloatDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatDevice) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    void backward(complexDoubleHost* data) {
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    void backward(complexFloatHost* data) {
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }
};

template <> inline const char* queryName<HQA2AGPU>() { return "HQA2AGPU"; }

template <class MPI_T, class FFTBackend> class HQPairGPU : public Backend {
  private:
    HQFFT::Distribution<HQFFT::PairSends, MPI_T, HQFFT::GPUReshape> dist;
    HQFFT::Dfft<HQFFT::Distribution, HQFFT::GPUReshape, HQFFT::PairSends, MPI_T,
                FFTBackend>
        dfft;

  public:
    HQPairGPU() {}

    HQPairGPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true)
        : dist(comm, ngx, ngx, ngx, blockSize), dfft(dist, ks_as_block) {}

    HQPairGPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize,
              bool ks_as_block = true)
        : dist(comm, ngx, ngy, ngz, blockSize), dfft(dist, ks_as_block) {}

    ~HQPairGPU(){};

    void query() {
        printf("Using HQPairSendsGPU\n");
        printf("   distribution = [%d %d %d]\n", dist.dims[0], dist.dims[1],
               dist.dims[2]);
    }

    int3 local_ng() { return dfft.local_ng(); }

    int local_ng(int i) { return dfft.local_ng(i); }

    int3 get_ks(int idx) { return dfft.get_ks(idx); }

    int3 get_rs(int idx) { return dfft.get_rs(idx); }

    int ngx() { return dfft.ng[0]; }

    int ngy() { return dfft.ng[1]; }

    int ngz() { return dfft.ng[2]; }

    int3 ng() { return make_int3(dfft.ng[0], dfft.ng[1], dfft.ng[2]); }

    int ng(int i) { return dfft.ng[i]; }

    size_t buff_sz() { return dist.nlocal; }

    int3 coords() {
        return make_int3(dist.coords[0], dist.coords[1], dist.coords[2]);
    }

    int3 dims() { return make_int3(dist.dims[0], dist.dims[1], dist.dims[2]); }

    int rank() { return dist.world_rank; }

    MPI_Comm comm() { return dist.world_comm; }

    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        dfft.forward(data, scratch);
    }

    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        dfft.forward(data, scratch);
    }

    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        dfft.backward(data, scratch);
    }

    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        dfft.backward(data, scratch);
    }

    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    void forward(complexDoubleDevice* data) {
        complexDoubleDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleDevice) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    void forward(complexFloatDevice* data) {
        complexFloatDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatDevice) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    void forward(complexDoubleHost* data) {
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    void forward(complexFloatHost* data) {
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    void backward(complexDoubleDevice* data) {
        complexDoubleDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleDevice) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    void backward(complexFloatDevice* data) {
        complexFloatDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatDevice) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    void backward(complexDoubleHost* data) {
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    void backward(complexFloatHost* data) {
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }
};

template <> inline const char* queryName<HQPairGPU>() { return "HQPairGPU"; }
#endif

template <class MPI_T, class FFTBackend> class HQA2ACPU : public Backend {
  private:
    HQFFT::Distribution<HQFFT::AllToAll, MPI_T, HQFFT::CPUReshape> dist;
    HQFFT::Dfft<HQFFT::Distribution, HQFFT::CPUReshape, HQFFT::AllToAll, MPI_T,
                FFTBackend>
        dfft;

  public:
    HQA2ACPU() {}

    HQA2ACPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true)
        : dist(comm, ngx, ngx, ngx, blockSize), dfft(dist, ks_as_block) {}

    HQA2ACPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize,
             bool ks_as_block = true)
        : dist(comm, ngx, ngy, ngz, blockSize), dfft(dist, ks_as_block) {}

    ~HQA2ACPU(){};

    void query() {
        printf("Using HQAllToAllCPU\n");
        printf("   distribution = [%d %d %d]\n", dist.dims[0], dist.dims[1],
               dist.dims[2]);
    }

    int3 local_ng() { return dfft.local_ng(); }

    int local_ng(int i) { return dfft.local_ng(i); }

    int3 get_ks(int idx) { return dfft.get_ks(idx); }

    int3 get_rs(int idx) { return dfft.get_rs(idx); }

    int ngx() { return dfft.ng[0]; }

    int ngy() { return dfft.ng[1]; }

    int ngz() { return dfft.ng[2]; }

    int3 ng() { return make_int3(dfft.ng[0], dfft.ng[1], dfft.ng[2]); }

    int ng(int i) { return dfft.ng[i]; }

    size_t buff_sz() { return dist.nlocal; }

    int3 coords() {
        return make_int3(dist.coords[0], dist.coords[1], dist.coords[2]);
    }

    int3 dims() { return make_int3(dist.dims[0], dist.dims[1], dist.dims[2]); }

    int rank() { return dist.world_rank; }

    MPI_Comm comm() { return dist.world_comm; }

    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        dfft.forward(data, scratch);
    }

    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        dfft.forward(data, scratch);
    }

#ifdef SWFFT_GPU
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.forward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }

    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.forward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }
#endif

    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        dfft.backward(data, scratch);
    }

    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        dfft.backward(data, scratch);
    }

#ifdef SWFFT_GPU
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.backward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }

    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.backward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }
#endif

    void forward(complexDoubleHost* data) {
        complexDoubleHost* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleHost) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    void forward(complexFloatHost* data) {
        complexFloatHost* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatHost) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

#ifdef SWFFT_GPU
    void forward(complexDoubleDevice* data) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.forward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }

    void forward(complexFloatDevice* data) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.forward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }
#endif

    void backward(complexDoubleHost* data) {
        complexDoubleHost* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleHost) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    void backward(complexFloatHost* data) {
        complexFloatHost* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatHost) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

#ifdef SWFFT_GPU
    void backward(complexDoubleDevice* data) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.backward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }

    void backward(complexFloatDevice* data) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        dfft.backward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }
#endif
};

template <> inline const char* queryName<HQA2ACPU>() { return "HQA2ACPU"; }

template <class MPI_T, class FFTBackend> class HQPairCPU : public Backend {
  private:
    HQFFT::Distribution<HQFFT::PairSends, MPI_T, HQFFT::CPUReshape> dist;
    HQFFT::Dfft<HQFFT::Distribution, HQFFT::CPUReshape, HQFFT::PairSends, MPI_T,
                FFTBackend>
        dfft;

  public:
    HQPairCPU() {}

    HQPairCPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true)
        : dist(comm, ngx, ngx, ngx, blockSize), dfft(dist, ks_as_block) {}

    HQPairCPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize,
              bool ks_as_block = true)
        : dist(comm, ngx, ngy, ngz, blockSize), dfft(dist, ks_as_block) {}

    ~HQPairCPU(){};

    void query() {
        printf("Using HQPairCPU\n");
        printf("   distribution = [%d %d %d]\n", dist.dims[0], dist.dims[1],
               dist.dims[2]);
    }

    int3 local_ng() { return dfft.local_ng(); }

    int local_ng(int i) { return dfft.local_ng(i); }

    int3 get_ks(int idx) { return dfft.get_ks(idx); }

    int3 get_rs(int idx) { return dfft.get_rs(idx); }

    int ngx() { return dfft.ng[0]; }

    int ngy() { return dfft.ng[1]; }

    int ngz() { return dfft.ng[2]; }

    int3 ng() { return make_int3(dfft.ng[0], dfft.ng[1], dfft.ng[2]); }

    int ng(int i) { return dfft.ng[i]; }

    size_t buff_sz() { return dist.nlocal; }

    int3 coords() {
        return make_int3(dist.coords[0], dist.coords[1], dist.coords[2]);
    }

    int3 dims() { return make_int3(dist.dims[0], dist.dims[1], dist.dims[2]); }

    int rank() { return dist.world_rank; }

    MPI_Comm comm() { return dist.world_comm; }

    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        dfft.forward(data, scratch);
    }

    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        dfft.forward(data, scratch);
    }

#ifdef SWFFT_GPU
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        forward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }

    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        forward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }
#endif

    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        dfft.backward(data, scratch);
    }

    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        dfft.backward(data, scratch);
    }

#ifdef SWFFT_GPU
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        backward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }

    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatHost* h_scratch;
        swfftAlloc(&h_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        backward(h_data, h_scratch);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        swfftFree(h_scratch);
    }
#endif

    void forward(complexDoubleHost* data) {
        complexDoubleHost* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleHost) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    void forward(complexFloatHost* data) {
        complexFloatHost* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatHost) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

#ifdef SWFFT_GPU
    void forward(complexDoubleDevice* data) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        // complexDoubleHost* h_scratch;
        // swfftAlloc(&h_scratch,sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        forward(h_data);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        // swfftFree(h_scratch);
    }

    void forward(complexFloatDevice* data) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        // complexFloatHost* h_scratch;
        // swfftAlloc(&h_scratch,sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        forward(h_data);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        // swfftFree(h_scratch);
    }
#endif

    void backward(complexDoubleHost* data) {
        complexDoubleHost* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleHost) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    void backward(complexFloatHost* data) {
        complexFloatHost* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatHost) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

#ifdef SWFFT_GPU
    void backward(complexDoubleDevice* data) {
        complexDoubleHost* h_data;
        swfftAlloc(&h_data, sizeof(complexDoubleDevice) * buff_sz());
        // complexDoubleHost* h_scratch;
        // swfftAlloc(&h_scratch,sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        backward(h_data);
        gpuMemcpy(data, h_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        // swfftFree(h_scratch);
    }

    void backward(complexFloatDevice* data) {
        complexFloatHost* h_data;
        swfftAlloc(&h_data, sizeof(complexFloatDevice) * buff_sz());
        // complexFloatHost* h_scratch;
        // swfftAlloc(&h_scratch,sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(h_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        backward(h_data);
        gpuMemcpy(data, h_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(h_data);
        // swfftFree(h_scratch);
    }
#endif
};

template <> inline const char* queryName<HQPairCPU>() { return "HQPairCPU"; }

} // namespace SWFFT

#endif // SWFFT_HQFFT
#endif // _SWFFT_HQFFT_HPP_