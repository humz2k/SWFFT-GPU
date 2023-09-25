#ifndef SWFFT_QUERY_SEEN
#define SWFFT_QUERY_SEEN

namespace SWFFT{

    template<template<class,class> class T>
    inline const char* queryName();

    template<class T>
    inline const char* queryName();

}

#endif