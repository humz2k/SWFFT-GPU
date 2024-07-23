/**
 * @file mpi_isend_irecv.hpp
 * @brief Header file for Isend and Irecv classes in the SWFFT namespace.
 */

#ifndef _SWFFT_ISEND_IRECV_HPP_
#define _SWFFT_ISEND_IRECV_HPP_

#include "complex-type.hpp"
#include "gpu.hpp"
#include "mpiwrangler.hpp"

namespace SWFFT {

/**
 * @class IsendBase
 * @brief Base class for non-blocking MPI send operations.
 *
 * @tparam XPUIsend_t Template for the send class.
 * @tparam T Data type of the buffer being sent.
 */
template <template <class> class XPUIsend_t, class T> class IsendBase {
  private:
    XPUIsend_t<T>* raw; /**< Pointer to the raw send object */

  public:
    /**
     * @brief Default constructor.
     */
    IsendBase(){};

    /**
     * @brief Parameterized constructor.
     *
     * @param in Pointer to the raw send object.
     */
    IsendBase(XPUIsend_t<T>* in) : raw(in){};

    /**
     * @brief Execute the non-blocking send operation.
     */
    void execute() { raw->execute(); };

    /**
     * @brief Wait for the non-blocking send operation to complete.
     */
    void wait() {
        raw->wait();
        delete raw;
    };
};

/**
 * @class IrecvBase
 * @brief Base class for non-blocking MPI receive operations.
 *
 * @tparam XPUIrecv_t Template for the receive class.
 * @tparam T Data type of the buffer being received.
 */
template <template <class> class XPUIrecv_t, class T> class IrecvBase {
  private:
    XPUIrecv_t<T>* raw; /**< Pointer to the raw receive object */

  public:
    /**
     * @brief Default constructor.
     */
    IrecvBase(){};

    /**
     * @brief Parameterized constructor.
     *
     * @param in Pointer to the raw receive object.
     */
    IrecvBase(XPUIrecv_t<T>* in) : raw(in){};

    /**
     * @brief Execute the non-blocking receive operation.
     */
    void execute() { raw->execute(); };

    /**
     * @brief Wait for the non-blocking receive operation to complete.
     */
    void wait() { raw->wait(); };

    /**
     * @brief Finalize the non-blocking receive operation.
     */
    void finalize() {
        raw->finalize();
        delete raw;
    };
};

/**
 * @class Isend
 * @brief Class template for non-blocking MPI send operations.
 *
 * @tparam MPI_T MPI implementation type.
 * @tparam T Data type of the buffer being sent.
 */
template <class MPI_T, class T> class Isend {};

/**
 * @class Irecv
 * @brief Class template for non-blocking MPI receive operations.
 *
 * @tparam MPI_T MPI implementation type.
 * @tparam T Data type of the buffer being received.
 */
template <class MPI_T, class T> class Irecv {};

#ifdef SWFFT_GPU
/**
 * @brief Specialization of Isend for double-precision complex numbers on the
 * GPU using CPUMPI.
 */
template <>
class Isend<CPUMPI, complexDoubleDevice>
    : public IsendBase<CPUIsend, complexDoubleDevice> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Isend for single-precision complex numbers on the
 * GPU using CPUMPI.
 */
template <>
class Isend<CPUMPI, complexFloatDevice>
    : public IsendBase<CPUIsend, complexFloatDevice> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Irecv for double-precision complex numbers on the
 * GPU using CPUMPI.
 */
template <>
class Irecv<CPUMPI, complexDoubleDevice>
    : public IrecvBase<CPUIrecv, complexDoubleDevice> {
  public:
    using IrecvBase::IrecvBase;
};

/**
 * @brief Specialization of Irecv for single-precision complex numbers on the
 * GPU using CPUMPI.
 */
template <>
class Irecv<CPUMPI, complexFloatDevice>
    : public IrecvBase<CPUIrecv, complexFloatDevice> {
  public:
    using IrecvBase::IrecvBase;
};
#endif

/**
 * @brief Specialization of Isend for double-precision complex numbers on the
 * CPU using CPUMPI.
 */
template <>
class Isend<CPUMPI, complexDoubleHost>
    : public IsendBase<CPUIsend, complexDoubleHost> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Isend for single-precision complex numbers on the
 * CPU using CPUMPI.
 */
template <>
class Isend<CPUMPI, complexFloatHost>
    : public IsendBase<CPUIsend, complexFloatHost> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Irecv for double-precision complex numbers on the
 * CPU using CPUMPI.
 */
template <>
class Irecv<CPUMPI, complexDoubleHost>
    : public IrecvBase<CPUIrecv, complexDoubleHost> {
  public:
    using IrecvBase::IrecvBase;
};

/**
 * @brief Specialization of Irecv for single-precision complex numbers on the
 * CPU using CPUMPI.
 */
template <>
class Irecv<CPUMPI, complexFloatHost>
    : public IrecvBase<CPUIrecv, complexFloatHost> {
  public:
    using IrecvBase::IrecvBase;
};

#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI
/**
 * @brief Specialization of Isend for double-precision complex numbers on the
 * GPU using GPUMPI.
 */
template <>
class Isend<GPUMPI, complexDoubleDevice>
    : public IsendBase<GPUIsend, complexDoubleDevice> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Isend for single-precision complex numbers on the
 * GPU using GPUMPI.
 */
template <>
class Isend<GPUMPI, complexFloatDevice>
    : public IsendBase<GPUIsend, complexFloatDevice> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Irecv for double-precision complex numbers on the
 * GPU using GPUMPI.
 */
template <>
class Irecv<GPUMPI, complexDoubleDevice>
    : public IrecvBase<GPUIrecv, complexDoubleDevice> {
  public:
    using IrecvBase::IrecvBase;
};

/**
 * @brief Specialization of Irecv for single-precision complex numbers on the
 * GPU using GPUMPI.
 */
template <>
class Irecv<GPUMPI, complexFloatDevice>
    : public IrecvBase<GPUIrecv, complexFloatDevice> {
  public:
    using IrecvBase::IrecvBase;
};

/**
 * @brief Specialization of Isend for double-precision complex numbers on the
 * CPU using GPUMPI.
 */
template <>
class Isend<GPUMPI, complexDoubleHost>
    : public IsendBase<GPUIsend, complexDoubleHost> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Isend for single-precision complex numbers on the
 * CPU using GPUMPI.
 */
template <>
class Isend<GPUMPI, complexFloatHost>
    : public IsendBase<GPUIsend, complexFloatHost> {
  public:
    using IsendBase::IsendBase;
};

/**
 * @brief Specialization of Irecv for double-precision complex numbers on the
 * CPU using GPUMPI.
 */
template <>
class Irecv<GPUMPI, complexDoubleHost>
    : public IrecvBase<GPUIrecv, complexDoubleHost> {
  public:
    using IrecvBase::IrecvBase;
};

/**
 * @brief Specialization of Irecv for single-precision complex numbers on the
 * CPU using GPUMPI.
 */
template <>
class Irecv<GPUMPI, complexFloatHost>
    : public IrecvBase<GPUIrecv, complexFloatHost> {
  public:
    using IrecvBase::IrecvBase;
};

#endif
#endif

}; // namespace SWFFT

#endif // _SWFFT_ISEND_IRECV_HPP_