/**
 * @file swfft.hpp
 * @brief Header file for the SWFFT library, providing FFT operations and
 * initialization functions.
 */

#ifndef _SWFFT_HPP_
#define _SWFFT_HPP_
#include "complex-type.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "gpu.hpp"
#include "mpi/mpiwrangler.hpp"
#include "timing-stats.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "query.hpp"

#ifdef SWFFT_ALLTOALL
#include "alltoall/alltoall.hpp"
#endif

#ifdef SWFFT_PAIRWISE
#include "pairwise/pairwise.hpp"
#endif

#ifdef SWFFT_HQFFT
#include "hqfft/hqfft.hpp"
#endif

namespace SWFFT {

/**
 * @brief Initialize threads for SWFFT.
 *
 * @param nthreads Number of threads to initialize. Default is 0.
 * @return int Number of threads initialized.
 */
inline int swfft_init_threads(int nthreads = 0) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef _OPENMP
    int omt = omp_get_max_threads();
#endif

#ifdef SWFFT_FFTW
    int out = 1;
#ifdef _OPENMP
    if (nthreads != 0) {
        out = swfft_fftw_init_threads(nthreads);
    } else {
        out = swfft_fftw_init_threads(omt);
    }

#endif
    if (rank == 0) {
        printf("swfft::fftw initialized with %d threads\n", out);
    }
    return out;
#endif

    return 0;
}

/**
 * @class swfft
 * @brief Main class for SWFFT operations, managing FFT operations and MPI
 * communication.
 *
 * @tparam DistBackend Distribution backend type.
 * @tparam MPI_T MPI implementation type.
 * @tparam FFTBackend FFT backend type.
 */
template <template <class, class> class DistBackend, class MPI_T,
          class FFTBackend>
class swfft {
  private:
    DistBackend<MPI_T, FFTBackend> backend; /**< Backend for FFT operations */
    double last_time; /**< Time taken for the last operation */
    int last_was; /**< Last operation performed: 0 for forward, 1 for backward
                   */

  public:
    /**
     * @brief Constructor for swfft with cubic grid.
     *
     * @param comm MPI communicator.
     * @param ngx Number of grid cells in each dimension.
     * @param blockSize Block size. Default is 64.
     * @param ks_as_block Flag indicating if k-space should be a block. Default
     * is true.
     */
    inline swfft(MPI_Comm comm, int ngx, int blockSize = 64,
                 bool ks_as_block = true)
        : backend(comm, ngx, blockSize, ks_as_block), last_time(0),
          last_was(-1) {}

    /**
     * @brief Constructor for swfft with non-cubic grid.
     *
     * @param comm MPI communicator.
     * @param ngx Number of grid cells in the x dimension.
     * @param ngy Number of grid cells in the y dimension.
     * @param ngz Number of grid cells in the z dimension.
     * @param blockSize Block size. Default is 64.
     * @param ks_as_block Flag indicating if k-space should be a block. Default
     * is true.
     */
    inline swfft(MPI_Comm comm, int ngx, int ngy, int ngz, int blockSize = 64,
                 bool ks_as_block = true)
        : backend(comm, ngx, ngy, ngz, blockSize, ks_as_block), last_time(0),
          last_was(-1) {}

    /**
     * @brief Destructor for swfft.
     */
    inline ~swfft() {}

    /**
     * @brief Print the time taken for the last operation.
     *
     * @return timing_stats_t Timing statistics for the last operation.
     */
    inline timing_stats_t printLastTime() {
        if (last_was == 0) {
            return printTimingStats(backend.comm(), "FORWARD ", last_time);
        } else {
            return printTimingStats(backend.comm(), "BACKWARD", last_time);
        }
    }

    /**
     * @brief Get the time taken for the last operation.
     *
     * @return timing_stats_t Timing statistics for the last operation.
     */
    inline timing_stats_t getLastTime() {
        return getTimingStats(backend.comm(), last_time);
    }

    /**
     * @brief Get the k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    inline int3 get_ks(int idx) { return backend.get_ks(idx); }

    /**
     * @brief Get the real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    inline int3 get_rs(int idx) { return backend.get_rs(idx); }

    /**
     * @brief Get the number of grid cells in the x dimension.
     *
     * @return int Number of grid cells in the x dimension.
     */
    inline int ngx() { return backend.ngx(); }

    /**
     * @brief Get the number of grid cells in the y dimension.
     *
     * @return int Number of grid cells in the y dimension.
     */
    inline int ngy() { return backend.ngy(); }

    /**
     * @brief Get the number of grid cells in the z dimension.
     *
     * @return int Number of grid cells in the z dimension.
     */
    inline int ngz() { return backend.ngz(); }

    /**
     * @brief Get the global size of the grid.
     *
     * @return int Global size of the grid.
     */
    inline int global_size() { return ngx() * ngy() * ngz(); }

    /**
     * @brief Get the number of grid cells in each dimension.
     *
     * @return int3 Number of grid cells in each dimension.
     */
    inline int3 ng() { return backend.ng(); }

    /**
     * @brief Get the number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Number of grid cells.
     */
    inline int ng(int i) { return backend.ng(i); }

    /**
     * @brief Get the local number of grid cells in each dimension.
     *
     * @return int3 Local number of grid cells.
     */
    inline int3 local_ng() { return backend.local_ng(); }

    /**
     * @brief Get the local number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    inline int local_ng(int i) { return backend.local_ng(i); }

    /**
     * @brief Get the local number of grid cells in the x dimension.
     *
     * @return int Local number of grid cells in the x dimension.
     */
    inline int local_ngx() { return backend.local_ng(0); }

    /**
     * @brief Get the local number of grid cells in the y dimension.
     *
     * @return int Local number of grid cells in the y dimension.
     */
    inline int local_ngy() { return backend.local_ng(1); }

    /**
     * @brief Get the local number of grid cells in the z dimension.
     *
     * @return int Local number of grid cells in the z dimension.
     */
    inline int local_ngz() { return backend.local_ng(2); }

    /**
     * @brief Get the buffer size required for FFT operations.
     *
     * @return size_t Buffer size.
     */
    inline size_t buff_sz() { return backend.buff_sz(); }

    /**
     * @brief Get the local size of the buffer.
     *
     * @return int Local size of the buffer.
     */
    inline int local_size() { return buff_sz(); }

    /**
     * @brief Get the coordinates of the current process.
     *
     * @return int3 Coordinates of the current process.
     */
    inline int3 coords() { return backend.coords(); }

    /**
     * @brief Get the dimensions of the process grid.
     *
     * @return int3 Dimensions of the process grid.
     */
    inline int3 dims() { return backend.dims(); }

    /**
     * @brief Get the rank of the current process.
     *
     * @return int Rank of the current process.
     */
    inline int rank() { return backend.rank(); }

    /**
     * @brief Get the MPI communicator.
     *
     * @return MPI_Comm MPI communicator.
     */
    inline MPI_Comm comm() { return backend.comm(); }

    /**
     * @brief Get the size of the MPI world.
     *
     * @return int Size of the MPI world.
     */
    inline int world_size() {
        int size;
        MPI_Comm_size(comm(), &size);
        return size;
    }

    /**
     * @brief Get the rank of the current process in the MPI world.
     *
     * @return int Rank of the current process.
     */
    inline int world_rank() { return rank(); }

    /**
     * @brief Query and print the parameters of the SWFFT setup.
     */
    inline void query() {
        if (!rank()) {
            printf("\n################\n");
            printf("SWFFT PARAMETERS\n");
            printf("   - DistBackend = %s\n", queryName<DistBackend>());
            printf("   - FFTBackend  = %s\n", queryName<FFTBackend>());
            printf("   - MPI_T       = %s\n", queryName<MPI_T>());
            printf("   - world_size  = %d\n", world_size());
            printf("   - dims        = [%d %d %d]\n", dims().x, dims().y,
                   dims().z);
            printf("   - ng          = [%d %d %d]\n", ngx(), ngy(), ngz());
            printf("   - local_ng    = [%d %d %d]\n", local_ngx(), local_ngy(),
                   local_ngz());
            printf("   - global_size = %d\n", global_size());
            printf("   - local_size  = %d\n", local_size());
            printf("   - buff_sz     = %ld\n", buff_sz());
            printf("################\n\n");
        }
    }

    /**
     * @brief Perform forward FFT transformation.
     *
     * @tparam T Data type of the buffer elements.
     * @param data input/output buffer.
     * @param scratch scratch buffer.
     */
    template <class T> inline void forward(T* data, T* scratch) {
        double start = MPI_Wtime();

        backend.forward(data, scratch);

        double end = MPI_Wtime();
        last_time = end - start;
        last_was = 0;
    }

    /**
     * @brief Perform backward FFT transformation.
     *
     * @tparam T Data type of the buffer elements.
     * @param data input/output buffer.
     * @param scratch scratch buffer.
     */
    template <class T> inline void backward(T* data, T* scratch) {
        double start = MPI_Wtime();

        backend.backward(data, scratch);

        double end = MPI_Wtime();
        last_time = end - start;
        last_was = 1;
    }

    /**
     * @brief Perform forward FFT transformation.
     *
     * @tparam T Data type of the buffer elements.
     * @param data input/output buffer.
     */
    template <class T> inline void forward(T* data) {
        double start = MPI_Wtime();

        backend.forward(data);

        double end = MPI_Wtime();
        last_time = end - start;
        last_was = 0;
    }

    /**
     * @brief Perform backward FFT transformation.
     *
     * @tparam T Data type of the buffer elements.
     * @param data input/output buffer.
     */
    template <class T> inline void backward(T* data) {
        double start = MPI_Wtime();

        backend.backward(data);

        double end = MPI_Wtime();
        last_time = end - start;
        last_was = 1;
    }
};

} // namespace SWFFT
#endif // _SWFFT_HPP_