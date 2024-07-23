/**
 * @file alltoall/dfft.hpp
 * @brief Header file for All-to-All distributed Fast Fourier Transform (DFFT)
 * operations in the SWFFT library.
 */

#ifndef _SWFFT_ALLTOALL_DFFT_HPP_
#define _SWFFT_ALLTOALL_DFFT_HPP_
#ifdef SWFFT_ALLTOALL

#include "alltoall/reorder.hpp"
#include "distribution.hpp"
#include "fftbackends/fftwrangler.hpp"
#include "logging.hpp"
#include "mpi/mpiwrangler.hpp"
#include "query.hpp"
#include "swfft_backend.hpp"
#include <mpi.h>

namespace SWFFT {
namespace A2A {
/**
 * @class Dfft
 * @brief Class for managing All-to-All distributed Fast Fourier Transform
 * (DFFT) operations.
 *
 * @tparam MPI_T MPI wrapper type.
 * @tparam REORDER_T Reordering strategy type.
 * @tparam FFTBackend FFT backend type.
 */
template <class MPI_T, class REORDER_T, class FFTBackend> class Dfft {
  private:
    /**
     * @brief Perform FFT operation.
     *
     * @tparam T Data type of the buffer.
     * @param data Input/output data buffer.
     * @param scratch Scratch buffer.
     * @param direction Direction of the FFT (forward or backward).
     */
    template <class T> void fft(T* data, T* scratch, fftdirection direction);
    bool ks_as_block; /**< Flag indicating if k-space should be a block */

  public:
    int ng[3];      /**< Number of grid cells in each dimension */
    int nlocal;     /**< Number of local grid cells */
    int world_size; /**< Size of the MPI world */
    int world_rank; /**< Rank of the current process */
    int blockSize;  /**< Block size for operations */

    Distribution<MPI_T, REORDER_T>& dist; /**< Distribution strategy instance */
    FFTBackend FFTs;                      /**< FFT backend instance */

    /**
     * @brief Constructor for Dfft.
     *
     * @param dist_ Distribution strategy instance.
     * @param ks_as_block_ Flag indicating if k-space should be treated as a
     * block.
     */
    Dfft(Distribution<MPI_T, REORDER_T>& dist_, bool ks_as_block_);

    /**
     * @brief Destructor for Dfft.
     */
    ~Dfft();

    /**
     * @brief Get k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    int3 get_ks(int idx);

    /**
     * @brief Get real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    int3 get_rs(int idx);

#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);

    /**
     * @brief Perform forward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
#endif
    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);

    /**
     * @brief Perform forward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void forward(complexFloatHost* Buff1, complexFloatHost* Buff2);

#ifdef SWFFT_GPU
    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexDoubleDevice* Buff1, complexDoubleDevice* Buff2);

    /**
     * @brief Perform backward FFT on device buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexFloatDevice* Buff1, complexFloatDevice* Buff2);
#endif
    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexDoubleHost* Buff1, complexDoubleHost* Buff2);

    /**
     * @brief Perform backward FFT on host buffers.
     *
     * @param Buff1 Input data buffer.
     * @param Buff2 Scratch buffer.
     */
    void backward(complexFloatHost* Buff1, complexFloatHost* Buff2);
};

} // namespace A2A
} // namespace SWFFT

#endif // SWFFT_ALLTOALL
#endif // _SWFFT_ALLTOALL_DFFT_HPP_