/**
 * @file swfft_backend.hpp
 * @brief Header file for backend interface class in the SWFFT namespace.
 */

#ifndef _SWFFT_BACKEND_HPP_
#define _SWFFT_BACKEND_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"
#include <mpi.h>

namespace SWFFT {

template<template<class, class> class T>
class dist3d_t{ };

/**
 * @class Backend
 * @brief Abstract base class for 3D FFT backend implementations.
 */
class Backend {
  public:
    /**
     * @brief Constructor for Backend.
     */
    Backend() {}

    /**
     * @brief Virtual destructor for Backend.
     */
    virtual ~Backend() {}

    /**
     * @brief Get the dimensions of the data grid.
     *
     * @return int3 Dimensions of the data grid.
     */
    virtual int3 dims() = 0;

    /**
     * @brief Get the local number of grid cells in each dimension.
     *
     * @return int3 Local number of grid cells in each dimension.
     */
    virtual int3 local_ng() = 0;

    /**
     * @brief Get the local number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Local number of grid cells.
     */
    virtual int local_ng(int i) = 0;

    /**
     * @brief Query the backend for information.
     */
    virtual void query() = 0;

    /**
     * @brief Get the k-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 k-space coordinates.
     */
    virtual int3 get_ks(int idx) = 0;

    /**
     * @brief Get the real-space coordinates for a given index.
     *
     * @param idx Index of the coordinate.
     * @return int3 Real-space coordinates.
     */
    virtual int3 get_rs(int idx) = 0;

    /**
     * @brief Get the number of grid cells in the x dimension.
     *
     * @return int Number of grid cells in the x dimension.
     */
    virtual int ngx() = 0;

    /**
     * @brief Get the number of grid cells in the y dimension.
     *
     * @return int Number of grid cells in the y dimension.
     */
    virtual int ngy() = 0;

    /**
     * @brief Get the number of grid cells in the z dimension.
     *
     * @return int Number of grid cells in the z dimension.
     */
    virtual int ngz() = 0;

    /**
     * @brief Get the number of grid cells in each dimension.
     *
     * @return int3 Number of grid cells in each dimension.
     */
    virtual int3 ng() = 0;

    /**
     * @brief Get the number of grid cells in a specific dimension.
     *
     * @param i Dimension index (0 for x, 1 for y, 2 for z).
     * @return int Number of grid cells.
     */
    virtual int ng(int i) = 0;

    /**
     * @brief Get the buffer size required for FFT operations.
     *
     * @return size_t Buffer size.
     */
    virtual size_t buff_sz() = 0;

    /**
     * @brief Get the coordinates of the current process.
     *
     * @return int3 Coordinates of the current process.
     */
    virtual int3 coords() = 0;

    /**
     * @brief Get the rank of the current process.
     *
     * @return int Rank of the current process.
     */
    virtual int rank() = 0;

    /**
     * @brief Get the MPI communicator.
     *
     * @return MPI_Comm MPI communicator.
     */
    virtual MPI_Comm comm() = 0;
#ifdef SWFFT_GPU
    /**
     * @brief Perform forward FFT on GPU data with double-precision complex data
     * and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void forward(complexDoubleDevice* data,
                         complexDoubleDevice* scratch) = 0;

    /**
     * @brief Perform forward FFT on GPU data with single-precision complex data
     * and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void forward(complexFloatDevice* data,
                         complexFloatDevice* scratch) = 0;

    /**
     * @brief Perform forward FFT on GPU data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void forward(complexDoubleDevice* data) = 0;

    /**
     * @brief Perform forward FFT on GPU data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void forward(complexFloatDevice* data) = 0;

    /**
     * @brief Perform backward FFT on GPU data with double-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void backward(complexDoubleDevice* data,
                          complexDoubleDevice* scratch) = 0;

    /**
     * @brief Perform backward FFT on GPU data with single-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void backward(complexFloatDevice* data,
                          complexFloatDevice* scratch) = 0;

    /**
     * @brief Perform backward FFT on GPU data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void backward(complexDoubleDevice* data) = 0;

    /**
     * @brief Perform backward FFT on GPU data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void backward(complexFloatDevice* data) = 0;
#endif
    /**
     * @brief Perform forward FFT on host data with double-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void forward(complexDoubleHost* data,
                         complexDoubleHost* scratch) = 0;

    /**
     * @brief Perform forward FFT on host data with single-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void forward(complexFloatHost* data, complexFloatHost* scratch) = 0;

    /**
     * @brief Perform forward FFT on host data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void forward(complexDoubleHost* data) = 0;

    /**
     * @brief Perform forward FFT on host data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void forward(complexFloatHost* data) = 0;

    /**
     * @brief Perform backward FFT on host data with double-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void backward(complexDoubleHost* data,
                          complexDoubleHost* scratch) = 0;

    /**
     * @brief Perform backward FFT on host data with single-precision complex
     * data and a scratch buffer.
     *
     * @param data Pointer to the input data buffer.
     * @param scratch Pointer to the scratch buffer.
     */
    virtual void backward(complexFloatHost* data,
                          complexFloatHost* scratch) = 0;

    /**
     * @brief Perform backward FFT on host data with double-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void backward(complexDoubleHost* data) = 0;

    /**
     * @brief Perform backward FFT on host data with single-precision complex
     * data.
     *
     * @param data Pointer to the input data buffer.
     */
    virtual void backward(complexFloatHost* data) = 0;
};

} // namespace SWFFT

#endif // _SWFFT_BACKEND_HPP_