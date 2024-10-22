/**
 * @file alltoall/alltoall.hpp
 * @brief Header file for All-to-All operations in the SWFFT library.
 */

#ifndef _SWFFT_ALLTOALL_HPP_
#define _SWFFT_ALLTOALL_HPP_
#ifdef SWFFT_ALLTOALL

#include "dfft.hpp"
#include "distribution.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "logging.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include "reorder.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>

namespace SWFFT {
template <class MPI_T, class FFTBackend> class AllToAllGPU;
template <class MPI_T, class FFTBackend> class AllToAllCPU;

#ifdef SWFFT_GPU
template <> class dist3d_t<AllToAllGPU> : public A2A::alltoallDist3d {
  public:
    using A2A::alltoallDist3d::alltoallDist3d;
    dist3d_t(A2A::alltoallDist3d in) : alltoallDist3d(in) {}
};
#endif

template <> class dist3d_t<AllToAllCPU> : public A2A::alltoallDist3d {
  public:
    using A2A::alltoallDist3d::alltoallDist3d;
    dist3d_t(A2A::alltoallDist3d in) : A2A::alltoallDist3d(in) {}
};

#ifdef SWFFT_GPU
/**
 * @class AllToAllGPU
 * @brief Class for GPU-based All-to-All FFT operations.
 *
 * @tparam MPI_T MPI wrapper type.
 * @tparam FFTBackend FFT backend type.
 */
template <class MPI_T, class FFTBackend> class AllToAllGPU : public Backend {
  private:
    A2A::Distribution<MPI_T, A2A::GPUReorder>
        m_dist; /**< Distribution strategy instance */
    A2A::Dfft<MPI_T, A2A::GPUReorder, FFTBackend> m_dfft; /**< DFFT instance */

  public:
    /**
     * @brief Default constructor for AllToAllGPU.
     */
    AllToAllGPU() {}

    /**
     * @brief Constructor for AllToAllGPU with cubic grid.
     *
     * @param comm MPI communicator.
     * @param ngx Number of grid cells in each dimension.
     * @param blockSize Block size for GPU operations.
     * @param ks_as_block Flag indicating if k-space should be a block. Default
     * is true.
     */
    AllToAllGPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true)
        : m_dist(comm, ngx, ngx, ngx, blockSize, ks_as_block),
          m_dfft(m_dist, ks_as_block) {}

    /**
     * @brief Constructor for AllToAllGPU with non-cubic grid.
     *
     * @param comm MPI communicator.
     * @param ngx Number of grid cells in the x dimension.
     * @param ngy Number of grid cells in the y dimension.
     * @param ngz Number of grid cells in the z dimension.
     * @param blockSize Block size for GPU operations.
     * @param ks_as_block Flag indicating if k-space should be a block. Default
     * is true.
     */
    AllToAllGPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize,
                bool ks_as_block = true)
        : m_dist(comm, ngx, ngy, ngz, blockSize, ks_as_block),
          m_dfft(m_dist, ks_as_block) {}

    /**
     * @brief Destructor for AllToAllGPU.
     */
    ~AllToAllGPU(){};

    /**
     * @brief Query and print the AllToAllGPU configuration.
     */
    void query() {
        printf("Using AllToAllGPU\n");
        printf("   distribution = [%d %d %d]\n", m_dist.dims[0], m_dist.dims[1],
               m_dist.dims[2]);
    }

    /**
     * @brief Get the local number of grid cells in each dimension.
     *
     * @return int3 Local number of grid cells.
     */
    int3 local_ng() {
        return make_int3(m_dist.local_grid_size[0], m_dist.local_grid_size[1],
                         m_dist.local_grid_size[2]);
    }

    /**
     * @brief Get the local number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng(int i) { return m_dist.local_grid_size[i]; }

    /**
     * @brief Gets a `dist3d_t`.
     *
     * This can be passed into GPU kernels (I think...), and has the methods
     *      `int3 dist3d_t::get_ks(int idx)`
     * and
     *      `int3 dist3d_t::get_rs(int idx)`
     */
    dist3d_t<AllToAllGPU> dist3d() {
        return dist3d_t<AllToAllGPU>(m_dfft.dist3d());
    }

    /**
     * @brief Get k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    int3 get_ks(int idx) { return m_dfft.get_ks(idx); }

    /**
     * @brief Get real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    int3 get_rs(int idx) { return m_dfft.get_rs(idx); }

    /**
     * @brief Get the number of grid cells in the x dimension.
     *
     * @return int Number of grid cells in the x dimension.
     */
    int ngx() { return m_dfft.ng[0]; }

    /**
     * @brief Get the number of grid cells in the y dimension.
     *
     * @return int Number of grid cells in the y dimension.
     */
    int ngy() { return m_dfft.ng[1]; }

    /**
     * @brief Get the number of grid cells in the z dimension.
     *
     * @return int Number of grid cells in the z dimension.
     */
    int ngz() { return m_dfft.ng[2]; }

    /**
     * @brief Get the number of grid cells in each dimension.
     *
     * @return int3 Number of grid cells in each dimension.
     */
    int3 ng() { return make_int3(m_dfft.ng[0], m_dfft.ng[1], m_dfft.ng[2]); }

    /**
     * @brief Get the number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Number of grid cells.
     */
    int ng(int i) { return m_dfft.ng[i]; }

    /**
     * @brief Get the buffer size required for FFT operations.
     *
     * @return size_t Buffer size.
     */
    size_t buff_sz() { return m_dist.nlocal; }

    /**
     * @brief Get the coordinates of the current process.
     *
     * @return int3 Coordinates of the current process.
     */
    int3 coords() {
        return make_int3(m_dist.coords[0], m_dist.coords[1], m_dist.coords[2]);
    }

    /**
     * @brief Get the dimensions of the process grid.
     *
     * @return int3 Dimensions of the process grid.
     */
    int3 dims() {
        return make_int3(m_dist.dims[0], m_dist.dims[1], m_dist.dims[2]);
    }

    /**
     * @brief Get the rank of the current process.
     *
     * @return int Rank of the current process.
     */
    int rank() { return m_dist.world_rank; }

    /**
     * @brief Get the MPI communicator.
     *
     * @return MPI_Comm MPI communicator.
     */
    MPI_Comm comm() { return m_dist.comm; }

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        m_dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        m_dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        m_dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        m_dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        m_dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        m_dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        complexDoubleDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        m_dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        complexFloatDevice* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatDevice) * buff_sz());
        complexFloatDevice* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        m_dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void forward(complexDoubleDevice* data) {
        complexDoubleDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleDevice) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void forward(complexFloatDevice* data) {
        complexFloatDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatDevice) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
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

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
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

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void backward(complexDoubleDevice* data) {
        complexDoubleDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleDevice) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void backward(complexFloatDevice* data) {
        complexFloatDevice* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatDevice) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
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

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
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

/**
 * @brief Query name for AllToAllGPU.
 *
 * @return const char* Name of the backend.
 */
template <> inline const char* queryName<AllToAllGPU>() {
    return "AllToAllGPU";
}
#endif

/**
 * @class AllToAllCPU
 * @brief Class for CPU-based All-to-All FFT operations.
 *
 * @tparam MPI_T MPI wrapper type.
 * @tparam FFTBackend FFT backend type.
 */
template <class MPI_T, class FFTBackend> class AllToAllCPU : public Backend {
  private:
    A2A::Distribution<MPI_T, A2A::CPUReorder>
        m_dist; /**< Distribution strategy instance */
    A2A::Dfft<MPI_T, A2A::CPUReorder, FFTBackend> m_dfft; /**< DFFT instance */

  public:
    /**
     * @brief Default constructor for AllToAllCPU.
     */
    AllToAllCPU() {}

    /**
     * @brief Constructor for AllToAllCPU with cubic grid.
     *
     * @param comm MPI communicator.
     * @param ngx Number of grid cells in each dimension.
     * @param blockSize Block size for operations.
     * @param ks_as_block Flag indicating if k-space should be a block. Default
     * is true.
     */
    AllToAllCPU(MPI_Comm comm, int ngx, int blockSize, bool ks_as_block = true)
        : m_dist(comm, ngx, ngx, ngx, blockSize, ks_as_block),
          m_dfft(m_dist, ks_as_block) {}

    /**
     * @brief Constructor for AllToAllCPU with non-cubic grid.
     *
     * @param comm MPI communicator.
     * @param ngx Number of grid cells in the x dimension.
     * @param ngy Number of grid cells in the y dimension.
     * @param ngz Number of grid cells in the z dimension.
     * @param blockSize Block size for operations.
     * @param ks_as_block Flag indicating if k-space should be a block. Default
     * is true.
     */
    AllToAllCPU(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize,
                bool ks_as_block = true)
        : m_dist(comm, ngx, ngy, ngz, blockSize, ks_as_block),
          m_dfft(m_dist, ks_as_block) {}

    /**
     * @brief Destructor for AllToAllCPU.
     */
    ~AllToAllCPU(){};

    /**
     * @brief Query and print the AllToAllCPU configuration.
     */
    void query() {
        printf("Using AllToAllCPU\n");
        printf("   distribution = [%d %d %d]\n", m_dist.dims[0], m_dist.dims[1],
               m_dist.dims[2]);
    }

    /**
     * @brief Get the local number of grid cells in each dimension.
     *
     * @return int3 Local number of grid cells.
     */
    int3 local_ng() {
        return make_int3(m_dist.local_grid_size[0], m_dist.local_grid_size[1],
                         m_dist.local_grid_size[2]);
    }

    /**
     * @brief Get the local number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng(int i) { return m_dist.local_grid_size[i]; }

    /**
     * @brief Gets a `dist3d_t`.
     *
     * This can be passed into GPU kernels (I think...), and has the methods
     *      `int3 dist3d_t::get_ks(int idx)`
     * and
     *      `int3 dist3d_t::get_rs(int idx)`
     */
    dist3d_t<AllToAllCPU> dist3d() {
        return dist3d_t<AllToAllCPU>(m_dfft.dist3d());
    }

    /**
     * @brief Get k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    int3 get_ks(int idx) { return m_dfft.get_ks(idx); }

    /**
     * @brief Get real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    int3 get_rs(int idx) { return m_dfft.get_rs(idx); }

    /**
     * @brief Get the number of grid cells in the x dimension.
     *
     * @return int Number of grid cells in the x dimension.
     */
    int ngx() { return m_dfft.ng[0]; }

    /**
     * @brief Get the number of grid cells in the y dimension.
     *
     * @return int Number of grid cells in the y dimension.
     */
    int ngy() { return m_dfft.ng[1]; }

    /**
     * @brief Get the number of grid cells in the z dimension.
     *
     * @return int Number of grid cells in the z dimension.
     */
    int ngz() { return m_dfft.ng[2]; }

    /**
     * @brief Get the number of grid cells in each dimension.
     *
     * @return int3 Number of grid cells in each dimension.
     */
    int3 ng() { return make_int3(m_dfft.ng[0], m_dfft.ng[1], m_dfft.ng[2]); }

    /**
     * @brief Get the number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Number of grid cells.
     */
    int ng(int i) { return m_dfft.ng[i]; }

    /**
     * @brief Get the buffer size required for FFT operations.
     *
     * @return size_t Buffer size.
     */
    size_t buff_sz() { return m_dist.nlocal; }

    /**
     * @brief Get the coordinates of the current process.
     *
     * @return int3 Coordinates of the current process.
     */
    int3 coords() {
        return make_int3(m_dist.coords[0], m_dist.coords[1], m_dist.coords[2]);
    }

    /**
     * @brief Get the dimensions of the process grid.
     *
     * @return int3 Dimensions of the process grid.
     */
    int3 dims() {
        return make_int3(m_dist.dims[0], m_dist.dims[1], m_dist.dims[2]);
    }

    /**
     * @brief Get the rank of the current process.
     *
     * @return int Rank of the current process.
     */
    int rank() { return m_dist.world_rank; }

    /**
     * @brief Get the MPI communicator.
     *
     * @return MPI_Comm MPI communicator.
     */
    MPI_Comm comm() { return m_dist.comm; }

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        m_dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        m_dfft.forward(data, scratch);
    }

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        complexDoubleHost* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        m_dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        complexFloatHost* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatHost) * buff_sz());
        complexFloatHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatHost) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        m_dfft.forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }
#endif
    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        m_dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        m_dfft.backward(data, scratch);
    }

#ifdef SWFFT_GPU
    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        complexDoubleHost* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        m_dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     */
    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        complexFloatHost* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatHost) * buff_sz());
        complexFloatHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatHost) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        m_dfft.backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_data);
        swfftFree(d_scratch);
    }
#endif
    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
    void forward(complexDoubleHost* data) {
        complexDoubleHost* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleHost) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
    void forward(complexFloatHost* data) {
        complexFloatHost* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatHost) * buff_sz());
        forward(data, scratch);
        swfftFree(scratch);
    }

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void forward(complexDoubleDevice* data) {
        complexDoubleHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void forward(complexFloatDevice* data) {
        complexFloatHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatHost) * buff_sz());
        complexFloatHost* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatHost) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        forward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }
#endif
    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
    void backward(complexDoubleHost* data) {
        complexDoubleHost* scratch;
        swfftAlloc(&scratch, sizeof(complexDoubleHost) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param data Input/output data buffer.
     */
    void backward(complexFloatHost* data) {
        complexFloatHost* scratch;
        swfftAlloc(&scratch, sizeof(complexFloatHost) * buff_sz());
        backward(data, scratch);
        swfftFree(scratch);
    }

#ifdef SWFFT_GPU
    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void backward(complexDoubleDevice* data) {
        complexDoubleHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexDoubleDevice) * buff_sz());
        complexDoubleHost* d_data;
        swfftAlloc(&d_data, sizeof(complexDoubleDevice) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexDoubleDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param data Input/output data buffer.
     */
    void backward(complexFloatDevice* data) {
        complexFloatHost* d_scratch;
        swfftAlloc(&d_scratch, sizeof(complexFloatHost) * buff_sz());
        complexFloatHost* d_data;
        swfftAlloc(&d_data, sizeof(complexFloatHost) * buff_sz());
        gpuMemcpy(d_data, data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyDeviceToHost);
        backward(d_data, d_scratch);
        gpuMemcpy(data, d_data, sizeof(complexFloatDevice) * buff_sz(),
                  gpuMemcpyHostToDevice);
        swfftFree(d_scratch);
        swfftFree(d_data);
    }
#endif
};

/**
 * @brief Query name for AllToAllCPU.
 *
 * @return const char* Name of the backend.
 */
template <> inline const char* queryName<AllToAllCPU>() {
    return "AllToAllCPU";
}

} // namespace SWFFT
#endif // SWFFT_ALLTOALL
#endif // _SWFFT_ALLTOALL_HPP_