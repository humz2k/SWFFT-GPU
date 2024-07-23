/**
 * @file pairwise/pairwise.hpp
 * @brief Header file for pairwise FFT operations in the SWFFT namespace.
 */

#ifndef _SWFFT_PAIRWISE_HPP_
#define _SWFFT_PAIRWISE_HPP_
#ifdef SWFFT_PAIRWISE

#include "fftbackends/fftwrangler.hpp"
#include "mpi/mpiwrangler.hpp"
#include "common.hpp"
#include "dfft.hpp"
#include "distribution.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>

namespace SWFFT {
/**
 * @class Pairwise
 * @brief Class to manage pairwise distributed FFT operations.
 *
 * @tparam MPI_T MPI implementation type (e.g., CPUMPI, GPUMPI).
 * @tparam FFTBackend FFT backend type (e.g., fftw, gpuFFT).
 */
template <class MPI_T, class FFTBackend> class Pairwise : public Backend {
  private:
    PAIR::Dfft<MPI_T, FFTBackend> dfft; /**< Distributed FFT manager */
    int n[3];                           /**< Dimensions of the data grid */
    int _buff_sz;                       /**< Buffer size */
    int _rank;                          /**< Rank of the current process */
    MPI_Comm _comm;                     /**< MPI communicator */

  public:
    /**
     * @brief Constructor for Pairwise with cubic grid.
     *
     * @param comm_ MPI communicator.
     * @param ngx Number of grid cells in each dimension.
     * @param blockSize Block size.
     * @param ks_as_block Flag indicating if k-space should be used as a block.
     */
    Pairwise(MPI_Comm comm_, int ngx, int blockSize, bool ks_as_block = true)
        : dfft(comm_, ngx, ngx, ngx), n{ngx, ngx, ngx}, _comm(comm_) {
        _buff_sz = dfft.buff_sz();
        MPI_Comm_rank(comm_, &_rank);
    }

    /**
     * @brief Constructor for Pairwise with non-cubic grid.
     *
     * @param comm_ MPI communicator.
     * @param ngx Number of grid cells in the x dimension.
     * @param ngy Number of grid cells in the y dimension.
     * @param ngz Number of grid cells in the z dimension.
     * @param blockSize Block size.
     * @param ks_as_block Flag indicating if k-space should be used as a block.
     */
    Pairwise(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize,
             bool ks_as_block = true)
        : dfft(comm_, ngx, ngy, ngz), n{ngx, ngy, ngz}, _comm(comm_) {
        _buff_sz = dfft.buff_sz();
        MPI_Comm_rank(comm_, &_rank);
    }

    /**
     * @brief Destructor for Pairwise.
     */
    ~Pairwise(){};

    /**
     * @brief Get the dimensions of the process grid.
     *
     * @return int3 Dimensions of the process grid.
     */
    int3 dims() {
        return make_int3(dfft.get_nproc_3d(0), dfft.get_nproc_3d(1),
                         dfft.get_nproc_3d(2));
    }

    /**
     * @brief Get the local number of grid cells in each dimension.
     *
     * @return int3 Local number of grid cells.
     */
    int3 local_ng() {
        return make_int3(dfft.get_local_ng_3d(0), dfft.get_local_ng_3d(1),
                         dfft.get_local_ng_3d(2));
    }

    /**
     * @brief Get the local number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    int local_ng(int i) { return dfft.get_local_ng_3d(i); }

    /**
     * @brief Query the backend for information.
     */
    void query() {
        printf("Using Pairwise\n");
        int3 my_dims = dims();
        printf("   distribution = [%d %d %d]\n", my_dims.x, my_dims.y,
               my_dims.z);
    }

    /**
     * @brief Get the k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    int3 get_ks(int idx) { return dfft.get_ks(idx); }

    /**
     * @brief Get the real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    int3 get_rs(int idx) { return dfft.get_rs(idx); }

    /**
     * @brief Get the number of grid cells in the x dimension.
     *
     * @return int Number of grid cells in the x dimension.
     */
    int ngx() { return n[0]; }

    /**
     * @brief Get the number of grid cells in the y dimension.
     *
     * @return int Number of grid cells in the y dimension.
     */
    int ngy() { return n[1]; }

    /**
     * @brief Get the number of grid cells in the z dimension.
     *
     * @return int Number of grid cells in the z dimension.
     */
    int ngz() { return n[2]; }

    /**
     * @brief Get the number of grid cells in each dimension.
     *
     * @return int3 Number of grid cells in each dimension.
     */
    int3 ng() { return make_int3(n[0], n[1], n[2]); }

    /**
     * @brief Get the number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Number of grid cells.
     */
    int ng(int i) { return n[i]; }

    /**
     * @brief Get the buffer size required for FFT operations.
     *
     * @return size_t Buffer size.
     */
    size_t buff_sz() { return _buff_sz; }

    /**
     * @brief Get the coordinates of the current process.
     *
     * @return int3 Coordinates of the current process.
     */
    int3 coords() { return dfft.coords(); }

    /**
     * @brief Get the rank of the current process.
     *
     * @return int Rank of the current process.
     */
    int rank() { return _rank; }

    /**
     * @brief Get the MPI communicator.
     *
     * @return MPI_Comm MPI communicator.
     */
    MPI_Comm comm() { return _comm; }

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on GPU data with double-precision complex data
     * and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        return dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on GPU data with single-precision complex data
     * and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexFloatDevice* data, complexFloatDevice* scratch) {
        return dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on GPU data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void forward(complexDoubleDevice* data) { return dfft.forward(data); }

    /**
     * @brief Perform forward FFT on GPU data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void forward(complexFloatDevice* data) { return dfft.forward(data); }

    /**
     * @brief Perform backward FFT on GPU data with double-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexDoubleDevice* data, complexDoubleDevice* scratch) {
        return dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on GPU data with single-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexFloatDevice* data, complexFloatDevice* scratch) {
        return dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on GPU data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void backward(complexDoubleDevice* data) { return dfft.backward(data); }

    /**
     * @brief Perform backward FFT on GPU data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void backward(complexFloatDevice* data) { return dfft.backward(data); }
#endif
    /**
     * @brief Perform forward FFT on host data with double-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexDoubleHost* data, complexDoubleHost* scratch) {
        return dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on host data with single-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void forward(complexFloatHost* data, complexFloatHost* scratch) {
        return dfft.forward(data, scratch);
    }

    /**
     * @brief Perform forward FFT on host data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void forward(complexDoubleHost* data) { return dfft.forward(data); }

    /**
     * @brief Perform forward FFT on host data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void forward(complexFloatHost* data) { return dfft.forward(data); }

    /**
     * @brief Perform backward FFT on host data with double-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexDoubleHost* data, complexDoubleHost* scratch) {
        return dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on host data with single-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    void backward(complexFloatHost* data, complexFloatHost* scratch) {
        return dfft.backward(data, scratch);
    }

    /**
     * @brief Perform backward FFT on host data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void backward(complexDoubleHost* data) { return dfft.backward(data); }

    /**
     * @brief Perform backward FFT on host data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    void backward(complexFloatHost* data) { return dfft.backward(data); }
};

/**
 * @brief Query the name of the Pairwise class.
 *
 * @return const char* The name of the Pairwise class.
 */
template <> inline const char* queryName<Pairwise>() { return "Pairwise"; }

} // namespace SWFFT
#endif // SWFFT_PAIRWISE
#endif // _SWFFT_PAIRWISE_HPP_