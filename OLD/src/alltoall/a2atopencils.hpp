#ifdef ALLTOALL
#include "gpu.hpp"
namespace A2A{
template<class T>
void launch_reorder_forward(T* src, T* dest, int mini_pencils_per_rank, int world_size, int mini_pencil_size, int blockSize, gpuStream_t stream);

template<class T>
void launch_reorder_backward(T* src, T* dest, int mini_pencils_per_rank, int world_size, int mini_pencil_size, int blockSize, gpuStream_t stream);
}
#endif
