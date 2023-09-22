#ifndef SWFFT_COMPLEXTYPE_H
#define SWFFT_COMPLEXTYPE_H

#include <stdlib.h>

namespace SWFFT{

typedef struct {
    double x;
    double y;
} complexDoubleHost;

typedef struct {
    float x;
    float y;
} complexFloatHost;

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

}

#endif
