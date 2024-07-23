#ifdef SWFFT_HQFFT

#include "hqfft/dfft.hpp"

namespace SWFFT {
namespace HQFFT {

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::Dfft(
    Dist<CollectiveComm, MPI_T, REORDER_T>& dist_, bool k_in_blocks_)
    : dist(dist_), ng{dist.ng[0], dist.ng[1], dist.ng[2]}, nlocal(dist.nlocal),
      k_in_blocks(k_in_blocks_) {}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::~Dfft() {}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int3 Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::get_ks(int idx) {
    if (k_in_blocks) {
        int3 local_idx;
        local_idx.x = idx / (dist.local_grid_size[1] * dist.local_grid_size[2]);
        local_idx.y = (idx - local_idx.x * (dist.local_grid_size[1] *
                                            dist.local_grid_size[2])) /
                      dist.local_grid_size[2];
        local_idx.z = (idx - local_idx.x * (dist.local_grid_size[1] *
                                            dist.local_grid_size[2])) -
                      (local_idx.y * dist.local_grid_size[2]);
        int3 global_idx = make_int3(dist.local_coords_start[0] + local_idx.x,
                                    dist.local_coords_start[1] + local_idx.y,
                                    dist.local_coords_start[2] + local_idx.z);
        return global_idx;
    } else {

        int z_dim = dist.ng[0];
        int x_dim = dist.local_grid_size[1] / dist.dims[0];
        int y_dim = (dist.nlocal / z_dim) / x_dim;

        int _x = idx / (y_dim * z_dim);
        int _y = (idx - (_x * y_dim * z_dim)) / z_dim;
        int _z = (idx - (_x * y_dim * z_dim)) - _y * z_dim;
        int new_idx = _z * x_dim * y_dim + _x * y_dim + _y;

        int dest_x_start = 0;

        int y = ((dist.coords[0] * dist.dims[2] + dist.coords[2]) *
                 (dist.ng[1] / (dist.dims[0] * dist.dims[2]))) /
                dist.local_grid_size[1];

        int y_send = dist.local_grid_size[1] /
                     (dist.ng[1] / (dist.dims[0] * dist.dims[2]));

        int z = (dist.coords[1] * (dist.ng[2] / dist.dims[1])) /
                dist.local_grid_size[2];

        int n_recvs = dist.dims[0];

        int count = new_idx / (dist.nlocal / n_recvs);

        int x = dest_x_start + count;

        int3 global_idx =
            make_int3(x * dist.local_grid_size[0], y * dist.local_grid_size[1],
                      z * dist.local_grid_size[2]);

        int ysrc = ((dist.coords[0] * dist.dims[2] + dist.coords[2]) *
                    (dist.ng[1] / (dist.dims[0] * dist.dims[2])));
        int zsrc = (dist.coords[1] * (dist.ng[2] / dist.dims[1]));

        int zoff = zsrc - dist.local_grid_size[2] * z;
        int yoff = (ysrc - dist.local_grid_size[1] * y) /
                   (dist.ng[1] / (dist.dims[0] * dist.dims[2]));
        int tmp1 = zoff / (dist.ng[2] / dist.dims[1]);
        int new_count = tmp1 * y_send + yoff;

        int i = (new_idx % (dist.nlocal / n_recvs)) +
                new_count * (dist.nlocal / n_recvs);

        int ny = y_send;
        int nz = n_recvs / y_send;

        {
            int3 local_dims =
                make_int3(dist.local_grid_size[0], dist.local_grid_size[1] / ny,
                          dist.local_grid_size[2] / nz); // per rank dims

            int n_recvs = ny * nz; // where we recieve from in each direction.
            int per_rank =
                nlocal / n_recvs;    // how many per rank we have recieved
            int rank = i / per_rank; // which rank I am from

            int i_local = i % per_rank; // my idx local to the rank I am from

            int3 local_coords;

            local_coords.x = i_local / (local_dims.y * local_dims.z);
            local_coords.y =
                (i_local - local_coords.x * local_dims.y * local_dims.z) /
                local_dims.z;
            local_coords.z =
                (i_local - local_coords.x * local_dims.y * local_dims.z) -
                local_coords.y * local_dims.z;

            int z_coord = rank / ny; // z is slow index for sends

            int y_coord = rank - z_coord * ny; // y is fast index for sends

            int z_offset = (dist.local_grid_size[2] / nz) * z_coord;

            int y_offset = (dist.local_grid_size[1] / ny) * y_coord;

            int3 global_coords =
                make_int3(local_coords.x, local_coords.y + y_offset,
                          local_coords.z + z_offset);
            global_idx.x += global_coords.x;
            global_idx.y += global_coords.y;
            global_idx.z += global_coords.z;
        }

        return global_idx;
    }
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int3 Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::get_rs(int idx) {
    int3 local_idx;
    local_idx.x = idx / (dist.local_grid_size[1] * dist.local_grid_size[2]);
    local_idx.y = (idx - local_idx.x * (dist.local_grid_size[1] *
                                        dist.local_grid_size[2])) /
                  dist.local_grid_size[2];
    local_idx.z = (idx - local_idx.x * (dist.local_grid_size[1] *
                                        dist.local_grid_size[2])) -
                  (local_idx.y * dist.local_grid_size[2]);
    int3 global_idx = make_int3(dist.local_coords_start[0] + local_idx.x,
                                dist.local_coords_start[1] + local_idx.y,
                                dist.local_coords_start[2] + local_idx.z);
    return global_idx;
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
template <class T>
inline void
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::_forward(T* buff1,
                                                                   T* buff2) {

    dist.pencils_1(buff1, buff2);

    FFTs.forward(buff1, buff2, ng[2], nlocal / ng[2]);

    dist.pencils_2(buff2, buff1);

    FFTs.forward(buff1, buff2, ng[1], nlocal / ng[1]);

    dist.pencils_3(buff2, buff1);

    FFTs.forward(buff1, buff2, ng[0], nlocal / ng[0]);

    if (k_in_blocks) {

        dist.return_pencils(buff1, buff2);

    } else {

        copyBuffers<T> cpy(buff1, buff2, nlocal);
        cpy.wait();
    }
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
template <class T>
inline void
Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::_backward(T* buff1,
                                                                    T* buff2) {
    // printf("???\n");
    if (k_in_blocks) {
        dist.pencils_1(buff1, buff2);

        FFTs.backward(buff1, buff2, ng[2], nlocal / ng[2]);

        dist.pencils_2(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[1], nlocal / ng[1]);

        dist.pencils_3(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[0], nlocal / ng[0]);

        dist.return_pencils(buff1, buff2);
    } else {

        FFTs.backward(buff1, buff2, ng[0], nlocal / ng[0]);

        dist.inverse_pencils_3(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[1], nlocal / ng[1]);

        dist.inverse_pencils_2(buff2, buff1);

        FFTs.backward(buff1, buff2, ng[2], nlocal / ng[2]);

        dist.inverse_pencils_1(buff2, buff1);

        copyBuffers<T> cpy(buff1, buff2, nlocal);
        cpy.wait();
    }
}

#ifdef SWFFT_GPU
template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexDoubleDevice* buff1, complexDoubleDevice* buff2) {
    _forward(buff1, buff2);
    gpuDeviceSynchronize();
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexFloatDevice* buff1, complexFloatDevice* buff2) {
    _forward(buff1, buff2);
    gpuDeviceSynchronize();
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexDoubleDevice* buff1, complexDoubleDevice* buff2) {
    _backward(buff1, buff2);
    gpuDeviceSynchronize();
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexFloatDevice* buff1, complexFloatDevice* buff2) {
    _backward(buff1, buff2);
    gpuDeviceSynchronize();
}
#endif

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexDoubleHost* buff1, complexDoubleHost* buff2) {
    _forward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::forward(
    complexFloatHost* buff1, complexFloatHost* buff2) {
    _forward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexDoubleHost* buff1, complexDoubleHost* buff2) {
    _backward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
void Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::backward(
    complexFloatHost* buff1, complexFloatHost* buff2) {
    _backward(buff1, buff2);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::buff_sz() {
    return nlocal;
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int3 Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::local_ng() {
    return make_int3(dist.local_grid_size[0], dist.local_grid_size[1],
                     dist.local_grid_size[2]);
}

template <template <template <class> class, class, class> class Dist,
          class REORDER_T, template <class> class CollectiveComm, class MPI_T,
          class FFTBackend>
int Dfft<Dist, REORDER_T, CollectiveComm, MPI_T, FFTBackend>::local_ng(int i) {
    return dist.local_grid_size[i];
}

#ifdef SWFFT_GPU
#ifdef SWFFT_CUFFT
template class Dfft<Distribution, GPUReshape, AllToAll, CPUMPI, gpuFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, CPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, AllToAll, CPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, CPUMPI, gpuFFT>;
#endif
#ifdef SWFFT_FFTW
template class Dfft<Distribution, GPUReshape, AllToAll, CPUMPI, fftw>;
template class Dfft<Distribution, GPUReshape, PairSends, CPUMPI, fftw>;
#endif
template class Dfft<Distribution, GPUReshape, AllToAll, CPUMPI, TestFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, CPUMPI, TestFFT>;
#endif

#ifdef SWFFT_FFTW
template class Dfft<Distribution, CPUReshape, AllToAll, CPUMPI, fftw>;
template class Dfft<Distribution, CPUReshape, PairSends, CPUMPI, fftw>;
#endif

template class Dfft<Distribution, CPUReshape, AllToAll, CPUMPI, TestFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, CPUMPI, TestFFT>;

#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI
#ifdef SWFFT_CUFFT
template class Dfft<Distribution, GPUReshape, AllToAll, GPUMPI, gpuFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, GPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, AllToAll, GPUMPI, gpuFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, GPUMPI, gpuFFT>;
#endif
#ifdef SWFFT_FFTW
template class Dfft<Distribution, GPUReshape, AllToAll, GPUMPI, fftw>;
template class Dfft<Distribution, GPUReshape, PairSends, GPUMPI, fftw>;
#endif
template class Dfft<Distribution, GPUReshape, AllToAll, GPUMPI, TestFFT>;
template class Dfft<Distribution, GPUReshape, PairSends, GPUMPI, TestFFT>;

#ifdef SWFFT_FFTW
template class Dfft<Distribution, CPUReshape, AllToAll, GPUMPI, fftw>;
template class Dfft<Distribution, CPUReshape, PairSends, GPUMPI, fftw>;
#endif

template class Dfft<Distribution, CPUReshape, AllToAll, GPUMPI, TestFFT>;
template class Dfft<Distribution, CPUReshape, PairSends, GPUMPI, TestFFT>;
#endif
#endif

} // namespace HQFFT
} // namespace SWFFT
#endif // SWFFT_HQFFT