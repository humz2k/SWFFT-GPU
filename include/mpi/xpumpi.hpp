/**
 * @file xpumpi.hpp
 * @brief Header file for templated MPI wrangling classes and functions in the
 * SWFFT namespace.
 */

#ifndef _SWFFT_XPUMPI_HPP_
#define _SWFFT_XPUMPI_HPP_

#include <mpi.h>

namespace SWFFT {

/**
 * @brief Perform an all-to-all communication using MPI.
 *
 * @tparam T Data type of the buffer elements.
 * @param buff1 Pointer to the first buffer.
 * @param buff2 Pointer to the second buffer.
 * @param n Number of elements per process.
 * @param comm MPI communicator.
 */
template <class T>
static inline void base_alltoall(T* buff1, T* buff2, int n, MPI_Comm comm) {
    MPI_Alltoall(buff1, n * sizeof(T), MPI_BYTE, buff2, n * sizeof(T), MPI_BYTE,
                 comm);
}

/**
 * @brief Perform a send/recv operation using MPI.
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
inline void base_sendrecv(T* send_buff, int sendcount, int dest, int sendtag,
                          T* recv_buff, int recvcount, int source, int recvtag,
                          MPI_Comm comm) {
    MPI_Sendrecv(send_buff, sendcount * sizeof(T), MPI_BYTE, dest, sendtag,
                 recv_buff, recvcount * sizeof(T), MPI_BYTE, source, recvtag,
                 comm, MPI_STATUS_IGNORE);
}

/**
 * @class XPUIsend
 * @brief Class for non-blocking MPI send operations.
 *
 * @tparam T Data type of the buffer being sent.
 */
template <class T> class XPUIsend {
  protected:
    bool initialized; /**< Initialization flag */
    T* in_buff;       /**< Input buffer */
    int n;            /**< Number of elements */
    int dest;         /**< Destination rank */
    int tag;          /**< MPI tag */
    MPI_Comm comm;    /**< MPI communicator */
    MPI_Request req;  /**< MPI request object */

  public:
    /**
     * @brief Default constructor.
     */
    XPUIsend() : initialized(false) {}

    /**
     * @brief Parameterized constructor.
     *
     * @param in_buff_ Pointer to the input buffer.
     * @param n_ Number of elements in the buffer.
     * @param dest_ Destination rank.
     * @param tag_ MPI tag.
     * @param comm_ MPI communicator.
     */
    XPUIsend(T* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_)
        : initialized(true), in_buff(in_buff_), n(n_), dest(dest_), tag(tag_),
          comm(comm_) {}

    /**
     * @brief Execute the non-blocking send operation.
     */
    void execute() {
        MPI_Isend(this->in_buff, this->n * sizeof(T), MPI_BYTE, this->dest,
                  this->tag, this->comm, &this->req);
    }

    /**
     * @brief Wait for the non-blocking send operation to complete.
     */
    void wait() { MPI_Wait(&this->req, MPI_STATUS_IGNORE); }
};

/**
 * @class XPUIrecv
 * @brief Class for non-blocking MPI receive operations.
 *
 * @tparam T Data type of the buffer being received.
 */
template <class T> class XPUIrecv {
  protected:
    bool initialized; /**< Initialization flag */
    T* out_buff;      /**< Output buffer */
    int n;            /**< Number of elements */
    int source;       /**< Source rank */
    int tag;          /**< MPI tag */
    MPI_Comm comm;    /**< MPI communicator */
    MPI_Request req;  /**< MPI request object */
    size_t sz;        /**< Size of the buffer */

  public:
    /**
     * @brief Default constructor.
     */
    XPUIrecv() : initialized(false) {}

    /**
     * @brief Parameterized constructor.
     *
     * @param out_buff_ Pointer to the output buffer.
     * @param n_ Number of elements in the buffer.
     * @param source_ Source rank.
     * @param tag_ MPI tag.
     * @param comm_ MPI communicator.
     */
    XPUIrecv(T* out_buff_, int n_, int source_, int tag_, MPI_Comm comm_)
        : initialized(true), out_buff(out_buff_), n(n_), source(source_),
          tag(tag_), comm(comm_) {}

    /**
     * @brief Execute the non-blocking receive operation.
     */
    void execute() {
        MPI_Irecv(this->out_buff, this->n * sizeof(T), MPI_BYTE, this->source,
                  this->tag, this->comm, &this->req);
    }

    /**
     * @brief Wait for the non-blocking receive operation to complete.
     */
    void wait() { MPI_Wait(&this->req, MPI_STATUS_IGNORE); }

    /**
     * @brief Finalize the non-blocking receive operation.
     */
    void finalize() {}
};

/**
 * @class XPUMPI
 * @brief Class for templated MPI operations.
 *
 * @tparam XPUIsend_t Template for the send class.
 * @tparam XPUIrecv_t Template for the receive class.
 */
template <template <class> class XPUIsend_t, template <class> class XPUIrecv_t>
class XPUMPI {
  public:
    /**
     * @brief Default constructor.
     */
    XPUMPI() {}

    /**
     * @brief Perform an all-to-all communication.
     *
     * @tparam T Data type of the buffer elements.
     * @param buff1 Pointer to the first buffer.
     * @param buff2 Pointer to the second buffer.
     * @param n Number of elements per process.
     * @param comm MPI communicator.
     */
    template <class T> void alltoall(T* buff1, T* buff2, int n, MPI_Comm comm) {
        base_alltoall(buff1, buff2, n, comm);
    }

    /**
     * @brief Perform a send/recv operation.
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
    void sendrecv(T* send_buff, int sendcount, int dest, int sendtag,
                  T* recv_buff, int recvcount, int source, int recvtag,
                  MPI_Comm comm) {
        base_sendrecv(send_buff, sendcount, dest, sendtag, recv_buff, recvcount,
                      source, recvtag, comm);
    }

    /**
     * @brief Perform a non-blocking send operation.
     *
     * @tparam T Data type of the buffer being sent.
     * @param buff Pointer to the buffer.
     * @param n Number of elements in the buffer.
     * @param dest Destination rank.
     * @param tag MPI tag.
     * @param comm MPI communicator.
     * @return XPUIsend_t<T>* Pointer to the XPUIsend object.
     */
    template <class T>
    XPUIsend_t<T>* isend(T* buff, int n, int dest, int tag, MPI_Comm comm) {
        return new XPUIsend_t<T>(buff, n, dest, tag, comm);
    }

    /**
     * @brief Perform a non-blocking receive operation.
     *
     * @tparam T Data type of the buffer being received.
     * @param buff Pointer to the buffer.
     * @param n Number of elements in the buffer.
     * @param source Source rank.
     * @param tag MPI tag.
     * @param comm MPI communicator.
     * @return XPUIrecv_t<T>* Pointer to the XPUIrecv object.
     */
    template <class T>
    XPUIrecv_t<T>* irecv(T* buff, int n, int source, int tag, MPI_Comm comm) {
        return new XPUIrecv_t<T>(buff, n, source, tag, comm);
    }
};

} // namespace SWFFT

#endif // _SWFFT_XPUMPI_HPP_