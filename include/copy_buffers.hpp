#ifndef SWFFT_COPYBUFFERS_SEEN
#define SWFFT_COPYBUFFERS_SEEN

#include "gpu.hpp"
#include "complex-type.h"

namespace SWFFT{

    template<class T>
    class copyBuffers{
        private:
            T* dest;
            T* src;
            int n;
            #ifdef SWFFT_GPU
            gpuEvent_t event;
            #endif
            
        public:
            inline copyBuffers(T* dest_, T* src_, int n_);
            inline ~copyBuffers();
            inline void wait();
    };

    template<class T>
    inline copyBuffers<T>::copyBuffers(T* dest_, T* src_, int n_) : dest(dest_), src(src_), n(n_){
        for (int i = 0; i < n; i++){
            dest[i] = src[i];
        }
    }

    template<class T>
    inline copyBuffers<T>::~copyBuffers(){

    }

    template<class T>
    inline void copyBuffers<T>::wait(){

    }
    #ifdef SWFFT_GPU

    template<>
    inline copyBuffers<complexDoubleDevice>::~copyBuffers(){

    }

    template<>
    inline copyBuffers<complexFloatDevice>::~copyBuffers(){

    }

    template<>
    inline copyBuffers<complexDoubleDevice>::copyBuffers(complexDoubleDevice* dest_, complexDoubleDevice* src_, int n_) : dest(dest_), src(src_), n(n_){
        gpuEventCreate(&event);
        gpuMemcpyAsync(dest,src,n * sizeof(complexDoubleDevice),gpuMemcpyDeviceToDevice);
        gpuEventRecord(event);
    }

    template<>
    inline void copyBuffers<complexDoubleDevice>::wait(){
        gpuEventSynchronize(event);
        gpuEventDestroy(event);
    }

    template<>
    inline copyBuffers<complexFloatDevice>::copyBuffers(complexFloatDevice* dest_, complexFloatDevice* src_, int n_) : dest(dest_), src(src_), n(n_){
        gpuEventCreate(&event);
        gpuMemcpyAsync(dest,src,n * sizeof(complexFloatDevice),gpuMemcpyDeviceToDevice);
        gpuEventRecord(event);
    }

    template<>
    inline void copyBuffers<complexFloatDevice>::wait(){
        gpuEventSynchronize(event);
        gpuEventDestroy(event);
    }
    #endif

}

#endif