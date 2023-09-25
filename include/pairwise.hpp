#ifdef SWFFT_PAIRWISE
#ifndef SWFFT_PAIRWISE_SEEN
#define SWFFT_PAIRWISE_SEEN

#include <mpi.h>
#include "fftwrangler.hpp"
#include "mpiwrangler.hpp"

namespace SWFFT{

namespace PAIR{

    typedef enum {
        REDISTRIBUTE_1_TO_3,
        REDISTRIBUTE_3_TO_1,
        REDISTRIBUTE_2_TO_3,
        REDISTRIBUTE_3_TO_2
    } redist_t;

    struct process_topology_t{
        MPI_Comm cart;
        int nproc[3];
        int period[3];
        int self[3];
        int n[3];
    };

    template<class T, class MPI_T>
    class distribution_t{

        public:
            bool debug;
            int n[3];

            MPI_T mpi;

            process_topology_t process_topology_1;
            process_topology_t process_topology_2_z;
            process_topology_t process_topology_2_y;
            process_topology_t process_topology_2_x;
            process_topology_t process_topology_3;

            T* d2_chunk;
            T* d3_chunk;
            MPI_Comm parent;

            distribution_t(MPI_Comm comm, int nx, int ny, int nz, bool debug_);
            /*distribution_t(MPI_Comm comm, const int n_[], 
                                int nproc_1d[],
                                int nproc_2d_x[],
                                int nproc_2d_y[],
                                int nproc_2d_z[],
                                int nproc_3d[],
                                bool debug_);*/
            ~distribution_t();

            void assert_commensurate();

            void redistribute(const T* a, T* b, redist_t r);

            void redistribute_2_and_3(const T* a, T* b, redist_t r, int z_dim);

            void redistribute_slab(const T* a, T* b, redist_t r);

            void dist_1_to_3(const T* a, T* b);

            void dist_3_to_1(const T* a, T* b);

            void dist_2_to_3(const T* a, T* b, int dim_z);

            void dist_3_to_2(const T* a, T* b, int dim_z);

            int get_nproc_1d(int direction);

            int get_nproc_2d_x(int direction);

            int get_nproc_2d_y(int direction);

            int get_nproc_2d_z(int direction);

            int get_nproc_3d(int direction);

            int get_self_1d(int direction);

            int get_self_2d_x(int direction);

            int get_self_2d_y(int direction);

            int get_self_2d_z(int direction);

            int get_self_3d(int direction);

            void coord_x_pencils(int myrank, int coord[]);

            void rank_x_pencils(int* myrank, int coord[]);

            int rank_x_pencils(int coord[]);

            void coord_y_pencils(int myrank, int coord[]);

            void rank_y_pencils(int* myrank, int coord[]);

            int rank_y_pencils(int coord[]);

            void coord_z_pencils(int myrank, int coord[]);

            void rank_z_pencils(int* myrank, int coord[]);

            int rank_z_pencils(int coord[]);

            void coord_cube(int myrank, int coord[]);

            void rank_cube(int* myrank, int coord[]);

            int rank_cube(int coord[]);

            int local_ng_1d(int i);
            int local_ng_2d_x(int i);
            int local_ng_2d_y(int i);
            int local_ng_2d_z(int i);
            int local_ng_3d(int i);

    };

    template<class MPI_T, class FFTBackend>
    class Dfft{
        public:

            Dfft(MPI_Comm comm_, int nx, int ny, int nz);
            ~Dfft();

            int buff_sz();
            int3 coords();
            int3 get_ks(int idx);
            int3 get_rs(int idx);
            int get_nproc_3d(int direction);

            #ifdef SWFFT_GPU
            void forward(complexDoubleDevice* data);
            void forward(complexFloatDevice* data);
            void backward(complexDoubleDevice* data);
            void backward(complexFloatDevice* data);

            void forward(complexDoubleDevice* data, complexDoubleDevice* scratch);
            void forward(complexFloatDevice* data, complexFloatDevice* scratch);
            void backward(complexDoubleDevice* data, complexDoubleDevice* scratch);
            void backward(complexFloatDevice* data, complexFloatDevice* scratch);
            #endif

            void forward(complexDoubleHost* data, complexDoubleHost* scratch);
            void forward(complexFloatHost* data, complexFloatHost* scratch);
            void backward(complexDoubleHost* data, complexDoubleHost* scratch);
            void backward(complexFloatHost* data, complexFloatHost* scratch);

            template<class T>
            void _forward(T* data);

            template<class T>
            void _backward(T* data);

            void forward(complexDoubleHost* data);
            void forward(complexFloatHost* data);
            void backward(complexDoubleHost* data);
            void backward(complexFloatHost* data);

        private:
            MPI_Comm comm;
            FFTBackend FFTs;
            int n[3];
            distribution_t<complexDoubleHost,MPI_T> double_dist;
            distribution_t<complexFloatHost,MPI_T> float_dist;

    };

}

template<class MPI_T,class FFTBackend>
class Pairwise{
    private:
        PAIR::Dfft<MPI_T,FFTBackend> dfft;
        int n[3];
        int _buff_sz;
        int _rank;
        MPI_Comm _comm;

    public:

        inline Pairwise(MPI_Comm comm_, int ngx, int blockSize, bool ks_as_block=true) : _comm(comm_), n{ngx,ngx,ngx}, dfft(comm_,ngx,ngx,ngx){
            _buff_sz = dfft.buff_sz();
            MPI_Comm_rank(comm_,&_rank);
        }

        inline Pairwise(MPI_Comm comm_, int ngx, int ngy, int ngz, int blockSize, bool ks_as_block=true) : _comm(comm_), n{ngx,ngy,ngz}, dfft(comm_,ngx,ngy,ngz){
            _buff_sz = dfft.buff_sz();
            MPI_Comm_rank(comm_,&_rank);
        }

        inline ~Pairwise(){};

        inline int3 dims(){
            return make_int3(dfft.get_nproc_3d(0),dfft.get_nproc_3d(1),dfft.get_nproc_3d(2));
        }

        inline void set_nsends(int x){
            
        }

        inline void set_delegate(int r){
            
        }

        inline void synchronize(){
            
        }

        inline void query(){
            printf("Using Pairwise\n");
            int3 my_dims = dims();
            printf("   distribution = [%d %d %d]\n",my_dims.x,my_dims.y,my_dims.z);
        }

        inline int3 get_ks(int idx){
            return dfft.get_ks(idx);
        }

        inline int3 get_rs(int idx){
            return dfft.get_rs(idx);
        }

        inline bool test_distribution(){
            return false;
        }

        inline int ngx(){
            return n[0];
        }

        inline int ngy(){
            return n[1];
        }

        inline int ngz(){
            return n[2];
        }

        inline int3 ng(){
            return make_int3(n[0],n[1],n[2]);
        }

        inline int ng(int i){
            return n[i];
        }

        inline int buff_sz(){
            return _buff_sz;
        }

        inline int3 coords(){
            return dfft.coords();
        }

        inline int rank(){
            return _rank;
        }

        inline MPI_Comm comm(){
            return _comm;
        }

        #ifdef SWFFT_GPU
        inline void forward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            return dfft.forward(data,scratch);
        }

        inline void forward(complexFloatDevice* data, complexFloatDevice* scratch){
            return dfft.forward(data,scratch);
        }

        inline void forward(complexDoubleDevice* data){
            return dfft.forward(data);
        }

        inline void forward(complexFloatDevice* data){
            return dfft.forward(data);
        }

        inline void backward(complexDoubleDevice* data, complexDoubleDevice* scratch){
            return dfft.backward(data,scratch);
        }

        inline void backward(complexFloatDevice* data, complexFloatDevice* scratch){
            return dfft.backward(data,scratch);
        }

        inline void backward(complexDoubleDevice* data){
            return dfft.backward(data);
        }

        inline void backward(complexFloatDevice* data){
            return dfft.backward(data);
        }

        #endif

        inline void forward(complexDoubleHost* data, complexDoubleHost* scratch){
            return dfft.forward(data,scratch);
        }

        inline void forward(complexFloatHost* data, complexFloatHost* scratch){
            return dfft.forward(data,scratch);
        }

        inline void forward(complexDoubleHost* data){
            return dfft.forward(data);
        }

        inline void forward(complexFloatHost* data){
            return dfft.forward(data);
        }

        inline void backward(complexDoubleHost* data, complexDoubleHost* scratch){
            return dfft.backward(data,scratch);
        }

        inline void backward(complexFloatHost* data, complexFloatHost* scratch){
            return dfft.backward(data,scratch);
        }

        inline void backward(complexDoubleHost* data){
            return dfft.backward(data);
        }

        inline void backward(complexFloatHost* data){
            return dfft.backward(data);
        }
};

#endif
}
#endif