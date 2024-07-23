/**
 * @file complex-type.hpp
 * @brief Header file for CPU complex number types and memory allocation
 * functions in the SWFFT namespace.
 */

#ifndef _SWFFT_COMPLEXTYPE_HPP_
#define _SWFFT_COMPLEXTYPE_HPP_

#include <stdlib.h>

namespace SWFFT {

/**
 * @struct complexDoubleHost
 * @brief A structure representing a complex number on the CPU with double
 * precision.
 */
struct complexDoubleHost {
    double x; /**< Real part of the complex number */
    double y; /**< Imaginary part of the complex number */
};

/**
 * @struct complexFloatHost
 * @brief A structure representing a complex number on the CPU with single
 * precision.
 */
struct complexFloatHost {
    float x; /**< Real part of the complex number */
    float y; /**< Imaginary part of the complex number */
};

/**
 * @brief Allocates memory for a CPU array of complexDoubleHost structures.
 *
 * @param ptr A pointer to the pointer that will hold the address of the
 * allocated memory.
 * @param sz The size of the memory to allocate.
 */
static inline void swfftAlloc(complexDoubleHost** ptr, size_t sz) {
    *ptr = (complexDoubleHost*)malloc(sz);
}

/**
 * @brief Allocates memory for a CPU array of complexFloatHost structures.
 *
 * @param ptr A pointer to the pointer that will hold the address of the
 * allocated memory.
 * @param sz The size of the memory to allocate.
 */
static inline void swfftAlloc(complexFloatHost** ptr, size_t sz) {
    *ptr = (complexFloatHost*)malloc(sz);
}

/**
 * @brief Frees the memory allocated for a CPU array of complexDoubleHost
 * structures.
 *
 * @param ptr A pointer to the memory to free.
 */
static inline void swfftFree(complexDoubleHost* ptr) { free(ptr); }

/**
 * @brief Frees the memory allocated for a CPU array of complexFloatHost
 * structures.
 *
 * @param ptr A pointer to the memory to free.
 */
static inline void swfftFree(complexFloatHost* ptr) { free(ptr); }

} // namespace SWFFT

#endif // _SWFFT_COMPLEXTYPE_HPP_
