/**
 * @file hqfft/collectivecomm.hpp
 * @brief Header file for collective communication classes in the SWFFT::HQFFT
 * namespace.
 */

#ifndef _SWFFT_HQFFT_COLLECTIVECOMM_HPP_
#define _SWFFT_HQFFT_COLLECTIVECOMM_HPP_

#include "common/copy_buffers.hpp"
#include "complex-type.hpp"
#include "gpu.hpp"
#include "mpi/mpi_isend_irecv.hpp"
#include "mpi/mpiwrangler.hpp"

namespace SWFFT {
namespace HQFFT {
/**
 * @class CollectiveCommunicator
 * @brief Abstract base class for collective communication implementations.
 *
 * @tparam MPI_T MPI implementation type (e.g., CPUMPI, GPUMPI).
 */
template <class MPI_T> class CollectiveCommunicator {
  public:
    MPI_T mpi; /**< MPI implementation instance */

    /**
     * @brief Constructor for CollectiveCommunicator.
     */
    CollectiveCommunicator() {}

    /**
     * @brief Virtual destructor for CollectiveCommunicator.
     */
    virtual ~CollectiveCommunicator() {}

#ifdef SWFFT_GPU
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * data on the GPU.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    virtual void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest,
                          int n_recv, MPI_Comm comm) = 0;

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * data on the GPU.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    virtual void alltoall(complexFloatDevice* src, complexFloatDevice* dest,
                          int n_recv, MPI_Comm comm) = 0;
#endif
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * data on the host.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    virtual void alltoall(complexDoubleHost* src, complexDoubleHost* dest,
                          int n_recv, MPI_Comm comm) = 0;

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * data on the host.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    virtual void alltoall(complexFloatHost* src, complexFloatHost* dest,
                          int n_recv, MPI_Comm comm) = 0;

    /**
     * @brief Query the collective communicator for information.
     */
    virtual void query() = 0;
};

/**
 * @class AllToAll
 * @brief Implementation of all-to-all communication using MPI_Alltoall.
 *
 * @tparam MPI_T MPI implementation type (e.g., CPUMPI, GPUMPI).
 */
template <class MPI_T> class AllToAll : public CollectiveCommunicator<MPI_T> {
  private:
    /**
     * @brief Internal all-to-all communication implementation.
     *
     * @tparam T Data type of the buffer elements.
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    template <class T>
    void _alltoall(T* src, T* dest, int n_recv, MPI_Comm comm) {
        this->mpi.alltoall(src, dest, n_recv, comm);
    }

  public:
#ifdef SWFFT_GPU
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * data on the GPU.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest,
                  int n_recv, MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * data on the GPU.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv,
                  MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }
#endif
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * data on the host.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv,
                  MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * data on the host.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv,
                  MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }

    /**
     * @brief Query the collective communicator for information.
     */
    void query() { printf("CollectiveCommunicator=AllToAll\n"); }
};

/**
 * @class PairSends
 * @brief Implementation of all-to-all communication using pairwise sends and
 * receives.
 *
 * @tparam MPI_T MPI implementation type (e.g., CPUMPI, GPUMPI).
 */
template <class MPI_T> class PairSends : public CollectiveCommunicator<MPI_T> {
  private:
    /**
     * @brief Internal all-to-all communication implementation using pairwise
     * sends and receives.
     *
     * @tparam T Data type of the buffer elements.
     * @param src_buff Source buffer.
     * @param dest_buff Destination buffer.
     * @param n Number of elements to send/receive.
     * @param comm MPI communicator.
     */
    template <class T>
    void _alltoall(T* src_buff, T* dest_buff, int n, MPI_Comm comm) {
        int comm_rank;
        MPI_Comm_rank(comm, &comm_rank);
        int comm_size;
        MPI_Comm_size(comm, &comm_size);

        copyBuffers<T> cpy(&dest_buff[comm_rank * n], &src_buff[comm_rank * n],
                           n);

        if (comm_size == 1) {
            cpy.wait();
            return;
        }
        if (comm_size == 2) {
            this->mpi.sendrecv(&src_buff[((comm_rank + 1) % comm_size) * n], n,
                               (comm_rank + 1) % comm_size, 0,
                               &dest_buff[((comm_rank + 1) % comm_size) * n], n,
                               (comm_rank + 1) % comm_size, 0, comm);
        } else {
            Isend<MPI_T, T>* sends =
                (Isend<MPI_T, T>*)malloc(sizeof(Isend<MPI_T, T>) * comm_size);
            Irecv<MPI_T, T>* recvs =
                (Irecv<MPI_T, T>*)malloc(sizeof(Irecv<MPI_T, T>) * comm_size);

            for (int i = 0; i < comm_size; i++) {
                if (i == comm_rank)
                    continue;
                sends[i] = Isend<MPI_T, T>(
                    this->mpi.isend(&src_buff[i * n], n, i, 0, comm));
                recvs[i] = Irecv<MPI_T, T>(
                    this->mpi.irecv(&dest_buff[i * n], n, i, 0, comm));
            }
            for (int i = 0; i < comm_size; i++) {
                if (i == comm_rank)
                    continue;
                sends[i].execute();
                recvs[i].execute();
            }

            for (int i = 0; i < comm_size; i++) {
                if (i == comm_rank)
                    continue;
                sends[i].wait();
                recvs[i].wait();
            }

            for (int i = 0; i < comm_size; i++) {
                if (i == comm_rank)
                    continue;
                recvs[i].finalize();
            }

            free(sends);
            free(recvs);
        }

        cpy.wait();
    }

  public:
#ifdef SWFFT_GPU
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * data on the GPU.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexDoubleDevice* src, complexDoubleDevice* dest,
                  int n_recv, MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * data on the GPU.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexFloatDevice* src, complexFloatDevice* dest, int n_recv,
                  MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }
#endif
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * data on the host.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexDoubleHost* src, complexDoubleHost* dest, int n_recv,
                  MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * data on the host.
     *
     * @param src Source buffer.
     * @param dest Destination buffer.
     * @param n_recv Number of elements to receive.
     * @param comm MPI communicator.
     */
    void alltoall(complexFloatHost* src, complexFloatHost* dest, int n_recv,
                  MPI_Comm comm) {
        _alltoall(src, dest, n_recv, comm);
    }

    /**
     * @brief Query the collective communicator for information.
     */
    void query() { printf("CollectiveCommunicator=PairSends\n"); }
};

} // namespace HQFFT
} // namespace SWFFT
#endif // _SWFFT_HQFFT_COLLECTIVECOMM_HPP_