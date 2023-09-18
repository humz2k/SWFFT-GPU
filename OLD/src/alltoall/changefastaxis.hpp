#ifdef ALLTOALL
#include "gpu.hpp"
namespace A2A{
template<class T>
void launch_fast_z_to_x(T* source, T* dest, int* local_grid_size, int blockSize, int numBlocks, int nlocal);

template<class T>
void launch_fast_x_to_y(T* source, T* dest, int* local_grid_size, int blockSize, int numBlocks, int nlocal);

template<class T>
void launch_fast_y_to_z(T* source, T* dest, int* local_grid_size, int blockSize, int numBlocks, int nlocal);
}
#endif
