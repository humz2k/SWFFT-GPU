/**
 * @file cpumpi.hpp
 * @brief Header file for CPUMPI wrangling classes and functions in the SWFFT
 * namespace.
 */

#ifndef _SWFFT_CPUMPI_HPP_
#define _SWFFT_CPUMPI_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"
#include "query.hpp"
#include "xpumpi.hpp"
#include <mpi.h>

namespace SWFFT {

#ifdef SWFFT_GPU
/**
 * @class CPUIsendGPUMemcpy
 * @brief Class for non-blocking MPI send operations with GPU memory copy.
 *
 * @tparam T Data type of the buffer being sent.
 */
template <class T> class CPUIsendGPUMemcpy : public XPUIsend<T> {
  private:
    void* h_in_buff;  /**< Host buffer pointer */
    gpuEvent_t event; /**< GPU event */

  public:
    /**
     * @brief Default constructor.
     */
    CPUIsendGPUMemcpy() : XPUIsend<T>() {}

    /**
     * @brief Parameterized constructor.
     *
     * @param in_buff_ Pointer to the input buffer.
     * @param n_ Number of elements in the buffer.
     * @param dest_ Destination rank.
     * @param tag_ MPI tag.
     * @param comm_ MPI communicator.
     */
    CPUIsendGPUMemcpy(T* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_)
        : XPUIsend<T>(in_buff_, n_, dest_, tag_, comm_) {
        size_t sz = sizeof(T) * this->n;
        this->h_in_buff = malloc(sz);
        gpuEventCreate(&event);
        gpuMemcpyAsync(this->h_in_buff, this->in_buff, sz,
                       gpuMemcpyDeviceToHost);
        gpuEventRecord(event);
    }

    /**
     * @brief Execute the non-blocking send operation.
     */
    void execute() {
        gpuEventSynchronize(event);
        MPI_Isend(this->h_in_buff, this->n * sizeof(T), MPI_BYTE, this->dest,
                  this->tag, this->comm, &this->req);
        gpuEventDestroy(event);
    }

    /**
     * @brief Wait for the non-blocking send operation to complete.
     */
    void wait() {
        MPI_Wait(&this->req, MPI_STATUS_IGNORE);
        free(this->h_in_buff);
    }
};
#endif

/**
 * @class CPUIsend
 * @brief Class for non-blocking MPI send operations.
 *
 * @tparam T Data type of the buffer being sent.
 */
template <class T> class CPUIsend : public XPUIsend<T> {
  public:
    using XPUIsend<T>::XPUIsend;
};

#ifdef SWFFT_GPU
/**
 * @class CPUIsend<complexDoubleDevice>
 * @brief Specialization of CPUIsend for double-precision complex numbers on the
 * GPU.
 */
template <>
class CPUIsend<complexDoubleDevice>
    : public CPUIsendGPUMemcpy<complexDoubleDevice> {
  public:
    using CPUIsendGPUMemcpy::CPUIsendGPUMemcpy;
};

/**
 * @class CPUIsend<complexFloatDevice>
 * @brief Specialization of CPUIsend for single-precision complex numbers on the
 * GPU.
 */
template <>
class CPUIsend<complexFloatDevice>
    : public CPUIsendGPUMemcpy<complexFloatDevice> {
  public:
    using CPUIsendGPUMemcpy::CPUIsendGPUMemcpy;
};
#endif

#ifdef SWFFT_GPU
/**
 * @class CPUIrecvGPUMemcpy
 * @brief Class for non-blocking MPI receive operations with GPU memory copy.
 *
 * @tparam T Data type of the buffer being received.
 */
template <class T> class CPUIrecvGPUMemcpy : public XPUIrecv<T> {
  private:
    void* h_out_buff; /**< Host buffer pointer */
    gpuEvent_t event; /**< GPU event */

  public:
    /**
     * @brief Default constructor.
     */
    CPUIrecvGPUMemcpy() : XPUIrecv<T>() {}

    /**
     * @brief Parameterized constructor.
     *
     * @param my_out_buff Pointer to the output buffer.
     * @param n Number of elements in the buffer.
     * @param source Source rank.
     * @param tag MPI tag.
     * @param comm MPI communicator.
     */
    CPUIrecvGPUMemcpy(T* my_out_buff, int n, int source, int tag, MPI_Comm comm)
        : XPUIrecv<T>(my_out_buff, n, source, tag, comm) {
        this->sz = this->n * sizeof(T);
        this->h_out_buff = malloc(this->sz);
    }

    /**
     * @brief Execute the non-blocking receive operation.
     */
    void execute() {
        MPI_Irecv(h_out_buff, this->sz, MPI_BYTE, this->source, this->tag,
                  this->comm, &this->req);
    }

    /**
     * @brief Wait for the non-blocking receive operation to complete.
     */
    void wait() {
        MPI_Wait(&this->req, MPI_STATUS_IGNORE);
        gpuEventCreate(&event);
        gpuMemcpyAsync(this->out_buff, h_out_buff, this->sz,
                       gpuMemcpyHostToDevice);
        gpuEventRecord(event);
    }

    /**
     * @brief Finalize the non-blocking receive operation.
     */
    void finalize() {
        gpuEventSynchronize(event);
        free(h_out_buff);
        gpuEventDestroy(event);
    }
};

#endif

/**
 * @class CPUIrecv
 * @brief Class for non-blocking MPI receive operations.
 *
 * @tparam T Data type of the buffer being received.
 */
template <class T> class CPUIrecv : public XPUIrecv<T> {
  public:
    using XPUIrecv<T>::XPUIrecv;
};

#ifdef SWFFT_GPU
/**
 * @class CPUIrecv<complexFloatDevice>
 * @brief Specialization of CPUIrecv for single-precision complex numbers on the
 * GPU.
 */
template <>
class CPUIrecv<complexFloatDevice>
    : public CPUIrecvGPUMemcpy<complexFloatDevice> {
  public:
    using CPUIrecvGPUMemcpy::CPUIrecvGPUMemcpy;
};

/**
 * @class CPUIrecv<complexDoubleDevice>
 * @brief Specialization of CPUIrecv for double-precision complex numbers on the
 * GPU.
 */
template <>
class CPUIrecv<complexDoubleDevice>
    : public CPUIrecvGPUMemcpy<complexDoubleDevice> {
  public:
    using CPUIrecvGPUMemcpy::CPUIrecvGPUMemcpy;
};
#endif

/**
 * @class CPUMPI
 * @brief Class for CPU-based MPI operations.
 */
class CPUMPI : public XPUMPI<CPUIsend, CPUIrecv> {
  private:
    void* _h_buff1;   /**< Host buffer 1 */
    void* _h_buff2;   /**< Host buffer 2 */
    size_t last_size; /**< Last buffer size */

#ifdef SWFFT_GPU
    /**
     * @brief Get host buffer 1 with specified size.
     *
     * @param sz Size of the buffer.
     * @return void* Pointer to the buffer.
     */
    void* get_h_buff1(size_t sz) {
        if (last_size == 0) {
            _h_buff1 = malloc(sz);
            _h_buff2 = malloc(sz);
            last_size = sz;
            return _h_buff1;
        }
        if (last_size < sz) {
            _h_buff1 = realloc(_h_buff1, sz);
            _h_buff2 = realloc(_h_buff2, sz);
            last_size = sz;
            return _h_buff1;
        }
        return _h_buff1;
    }

    /**
     * @brief Get host buffer 2 with specified size.
     *
     * @param sz Size of the buffer.
     * @return void* Pointer to the buffer.
     */
    void* get_h_buff2(size_t sz) {
        get_h_buff1(sz);
        return _h_buff2;
    }

    /**
     * @brief Perform an all-to-all communication with GPU memory.
     *
     * @tparam T Data type of the buffer elements.
     * @param buff1 Pointer to the first buffer.
     * @param buff2 Pointer to the second buffer.
     * @param n Number of elements per process.
     * @param comm MPI communicator.
     */
    template <class T>
    void gpu_memcpy_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm) {
        int world_size;
        MPI_Comm_size(comm, &world_size);
        int sz = world_size * n * sizeof(T);
        T* h_buff1 = (T*)get_h_buff1(sz);
        T* h_buff2 = (T*)get_h_buff2(sz);
        gpuMemcpy(h_buff1, buff1, sz, gpuMemcpyDeviceToHost);
        base_alltoall(h_buff1, h_buff2, n, comm);
        gpuMemcpy(buff2, h_buff2, sz, gpuMemcpyHostToDevice);
    }

    /**
     * @brief Perform a send/recv operation with GPU memory.
     *
     * @tparam T Data type of the buffer elements.
     * @param send_buff Pointer to the send buffer.
     * @param sendcount Number of elements to send.
     * @param dest Destination rank.
     * @param sendtag MPI tag for the send.
     * @param recv_buff Pointer to the receive buffer.
     * @param recvcount Number of elements to receive.
     * @param source Source rank.
     * @param recvtag MPI tag for the receive.
     * @param comm MPI communicator.
     */
    template <class T>
    void gpu_memcpy_sendrecv(T* send_buff, int sendcount, int dest, int sendtag,
                             T* recv_buff, int recvcount, int source,
                             int recvtag, MPI_Comm comm) {
        size_t send_size = sendcount * sizeof(T);
        size_t recv_size = recvcount * sizeof(T);
        T* h_buff1 = (T*)get_h_buff1(send_size);
        T* h_buff2 = (T*)get_h_buff2(recv_size);
        gpuMemcpy(h_buff1, send_buff, send_size, gpuMemcpyDeviceToHost);
        base_sendrecv(h_buff1, sendcount, dest, sendtag, h_buff2, recvcount,
                      source, recvtag, comm);
        gpuMemcpy(recv_buff, h_buff2, recv_size, gpuMemcpyHostToDevice);
    }
#endif

  public:
    using XPUMPI::alltoall;
    using XPUMPI::sendrecv;

    /**
     * @brief Default constructor.
     */
    CPUMPI() : last_size(0) {}

    /**
     * @brief Destructor.
     */
    ~CPUMPI() {
        if (last_size != 0) {
            free(_h_buff1);
            free(_h_buff2);
        }
    }

    /**
     * @brief Query the MPI implementation being used.
     */
    void query() { printf("Using CPUMPI\n"); }

#ifdef SWFFT_GPU
    /**
     * @brief Perform an all-to-all communication for double-precision complex
     * numbers on the GPU.
     *
     * @param buff1 Pointer to the first buffer.
     * @param buff2 Pointer to the second buffer.
     * @param n Number of elements per process.
     * @param comm MPI communicator.
     */
    void alltoall(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n,
                  MPI_Comm comm) {
        gpu_memcpy_alltoall(buff1, buff2, n, comm);
    }

    /**
     * @brief Perform an all-to-all communication for single-precision complex
     * numbers on the GPU.
     *
     * @param buff1 Pointer to the first buffer.
     * @param buff2 Pointer to the second buffer.
     * @param n Number of elements per process.
     * @param comm MPI communicator.
     */
    void alltoall(complexFloatDevice* buff1, complexFloatDevice* buff2, int n,
                  MPI_Comm comm) {
        gpu_memcpy_alltoall(buff1, buff2, n, comm);
    }

    /**
     * @brief Perform a send/recv operation for double-precision complex numbers
     * on the GPU.
     *
     * @param send_buff Pointer to the send buffer.
     * @param sendcount Number of elements to send.
     * @param dest Destination rank.
     * @param sendtag MPI tag for the send.
     * @param recv_buff Pointer to the receive buffer.
     * @param recvcount Number of elements to receive.
     * @param source Source rank.
     * @param recvtag MPI tag for the receive.
     * @param comm MPI communicator.
     */
    void sendrecv(complexDoubleDevice* send_buff, int sendcount, int dest,
                  int sendtag, complexDoubleDevice* recv_buff, int recvcount,
                  int source, int recvtag, MPI_Comm comm) {
        gpu_memcpy_sendrecv(send_buff, sendcount, dest, sendtag, recv_buff,
                            recvcount, source, recvtag, comm);
    }

    /**
     * @brief Perform a send/recv operation for single-precision complex numbers
     * on the GPU.
     *
     * @param send_buff Pointer to the send buffer.
     * @param sendcount Number of elements to send.
     * @param dest Destination rank.
     * @param sendtag MPI tag for the send.
     * @param recv_buff Pointer to the receive buffer.
     * @param recvcount Number of elements to receive.
     * @param source Source rank.
     * @param recvtag MPI tag for the receive.
     * @param comm MPI communicator.
     */
    void sendrecv(complexFloatDevice* send_buff, int sendcount, int dest,
                  int sendtag, complexFloatDevice* recv_buff, int recvcount,
                  int source, int recvtag, MPI_Comm comm) {
        gpu_memcpy_sendrecv(send_buff, sendcount, dest, sendtag, recv_buff,
                            recvcount, source, recvtag, comm);
    }
#endif
};

/**
 * @brief Query the name of the CPUMPI class.
 *
 * @return const char* The name of the CPUMPI class.
 */
template <> inline const char* queryName<CPUMPI>() { return "CPUMPI"; }

} // namespace SWFFT

#endif // _SWFFT_CPUMPI_HPP_