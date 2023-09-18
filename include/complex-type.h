// Compatibility file for C99 and C++ complex.  This header
// can be included by either C99 or ANSI C++ programs to
// allow complex arithmetic to be written in a common subset.
// Note that overloads for both the real and complex math
// functions are available after this header has been
// included.

#ifndef HACC_COMPLEXTYPE_H
#define HACC_COMPLEXTYPE_H

typedef struct {
    double x;
    double y;
} complexDoubleHost;

typedef struct {
    float x;
    float y;
} complexFloatHost;

#include <stdlib.h>

inline void swfftAlloc(complexDoubleHost** ptr, size_t sz){
    *ptr = (complexDoubleHost*)malloc(sz);
}

inline void swfftAlloc(complexFloatHost** ptr, size_t sz){
    *ptr = (complexFloatHost*)malloc(sz);
}

inline void swfftFree(complexDoubleHost* ptr){
    free(ptr);
}

inline void swfftFree(complexFloatHost* ptr){
    free(ptr);
}

/*#ifdef __cplusplus

#include <cmath>
#include <complex>

typedef std::complex<double> complexDoubleHost;
typedef std::complex<float> complexFloatHost;

#define I complexDoubleHost(0.0, 1.0)

#else

#include <complex.h>
#include <math.h>

typedef double complex complexDoubleHost;
typedef float complex complexFloatHost;

#define complex_t(r,i) ((double)(r) + ((double)(i)) * I)

#define real(x) creal(x)
#define imag(x) cimag(x)
#define abs(x) fabs(x)
#define arg(x) carg(x)

#endif  // #ifdef __cplusplus*/

#endif  // HACC_COMPLEXTYPE_H
