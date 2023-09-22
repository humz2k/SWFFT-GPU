#ifdef SWFFT_HQFFT
#ifndef SWFFT_HQFFT_REORDER_SEEN
#define SWFFT_HQFFT_REORDER_SEEN

#include "gpu.hpp"
#include "complex-type.h"

namespace SWFFT{

namespace HQFFT{

    #ifdef SWFFT_GPU
    class GPUReshape{
        private:
            template<class T>
            inline void _reshape(T* buff1, T* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);

            template<class T>
            inline void _inverse_reshape(T* buff1, T* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);

            template<class T>
            inline void _unreshape(T* buff1, T* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);

            template<class T>
            inline void _inverse_unreshape(T* buff1, T* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);

            template<class T>
            inline void _reshape_final(T* buff1, T* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

        public:
            GPUReshape();
            ~GPUReshape();

            
            void reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

            void reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexFloatDevice* buff1, complexFloatDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

            void reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexDoubleHost* buff1, complexDoubleHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

            void reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexFloatHost* buff1, complexFloatHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

    };
    #endif
    class CPUReshape{
        private:
            template<class T>
            inline void _reshape(const T* __restrict buff1, T* __restrict buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);

            template<class T>
            inline void _unreshape(const T* __restrict buff1, T* __restrict buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);

            template<class T>
            inline void _inverse_reshape(const T* __restrict buff1, T* __restrict buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);

            template<class T>
            inline void _inverse_unreshape(const T* __restrict buff1, T* __restrict buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);

            template<class T>
            inline void _reshape_final(const T* __restrict buff1, T* __restrict buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

        public:
            CPUReshape();
            ~CPUReshape();

            #ifdef SWFFT_GPU
            void reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexDoubleDevice* buff1, complexDoubleDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

            void reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexFloatDevice* buff1, complexFloatDevice* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexFloatDevice* buff1, complexFloatDevice* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);
            #endif

            void reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexDoubleHost* buff1, complexDoubleHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexDoubleHost* buff1, complexDoubleHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

            void reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void inverse_reshape(complexFloatHost* buff1, complexFloatHost* buff2, int n_recvs, int mini_pencil_size, int send_per_rank, int pencils_per_rank, int nlocal, int blockSize);
            void inverse_unreshape(complexFloatHost* buff1, complexFloatHost* buff2, int z_dim, int x_dim, int y_dim, int nlocal, int blockSize);
            void reshape_final(complexFloatHost* buff1, complexFloatHost* buff2, int ny, int nz, int local_grid_size[], int nlocal, int blockSize);

    };

}

}

#endif
#endif