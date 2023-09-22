#ifdef SWFFT_PAIRWISE

#ifdef _OPENMP
#define DIST_OMP
#endif

#ifdef DIST_OMP
#include <omp.h>
#endif

#define PENCIL

#include "pairwise.hpp"
#include <cassert>

#ifndef USE_SLAB_WORKAROUND
#define USE_SLAB_WORKAROUND 0
#endif

#define DEBUG_CONDITION false

//#define PRINT_DISTRIBUTION
namespace SWFFT{
namespace PAIR{


    static inline const char *separator(int i, int n)
    {
        return i == (n - 1) ? "." : ", ";
    }

    template<class T, class MPI_T>
    distribution_t<T,MPI_T>::distribution_t(MPI_Comm comm, int nx, int ny, int nz, bool debug_) : parent(comm), n{nx,ny,nz}, debug(debug_){
        
        int nproc;
        int self;
        int ndim = 3;
        int period[3];

        MPI_Comm_rank(comm, &self);
        MPI_Comm_size(comm, &nproc);

        process_topology_1.nproc[0] = 0;
        process_topology_1.nproc[1] = 1;
        process_topology_1.nproc[2] = 1;
        period[0] = period[1] = period[2] = 1;

        MPI_Dims_create(nproc,ndim,process_topology_1.nproc);

        #ifdef PRINT_DISTRIBUTION
        if(self == 0) {
            printf("distribution 1D: [%d:%d:%d]\n",
            process_topology_1.nproc[0],
            process_topology_1.nproc[1],
            process_topology_1.nproc[2]);
            fflush(stdout);
        }
        #endif

        //creates the new communicator
        MPI_Cart_create(comm, ndim, process_topology_1.nproc, period, 0, 
                &process_topology_1.cart);
        //gets .self (is coordinate)
        MPI_Cart_get(process_topology_1.cart, ndim, process_topology_1.nproc, 
                process_topology_1.period, process_topology_1.self);
        //calculates the local dimensions (number of points in each dimension)
        process_topology_1.n[0] = n[0] / process_topology_1.nproc[0];
        process_topology_1.n[1] = n[1] / process_topology_1.nproc[1];
        process_topology_1.n[2] = n[2] / process_topology_1.nproc[2];


        // set up process grid with 3d decomposition (CUBE)
        process_topology_3.nproc[0] = 0;
        process_topology_3.nproc[1] = 0;
        process_topology_3.nproc[2] = 0;
        period[0] = period[1] = period[2] = 1;
        MPI_Dims_create(nproc, ndim, process_topology_3.nproc);

        #ifdef PRINT_DISTRIBUTION
        if(self == 0) {
            printf("distribution 3D: [%d:%d:%d]\n",
            process_topology_3.nproc[0],
            process_topology_3.nproc[1],
            process_topology_3.nproc[2]);
            fflush(stdout);
        }
        #endif

        MPI_Cart_create(comm, ndim, process_topology_3.nproc, period, 0, 
		  &process_topology_3.cart);
        //finds cartesian coordinate of this current rank
        coord_cube(self,process_topology_3.self);

        assert(n[0]%process_topology_3.nproc[0] == 0);
        assert(n[0]%process_topology_3.nproc[1] == 0);
        assert(n[0]%process_topology_3.nproc[2] == 0);

        // set up process grid with 2d decomposition (z_PENCILs )
        process_topology_3.n[0] = n[0] / process_topology_3.nproc[0];
        process_topology_3.n[1] = n[1] / process_topology_3.nproc[1];
        process_topology_3.n[2] = n[2] / process_topology_3.nproc[2];

        period[0] = period[1] = period[2] = 1;

        process_topology_2_z.nproc[0] = 0;
        process_topology_2_z.nproc[1] = 0;
        process_topology_2_z.nproc[2] = 1;

        MPI_Dims_create(nproc, ndim, process_topology_2_z.nproc);
        process_topology_2_z.n[0] = n[0] / process_topology_2_z.nproc[0];
        process_topology_2_z.n[1] = n[1] / process_topology_2_z.nproc[1];
        process_topology_2_z.n[2] = n[2] / process_topology_2_z.nproc[2];

        bool check_z_dims=false; 
        if(process_topology_2_z.n[0] != 0 
            && process_topology_2_z.n[1] != 0 
            && process_topology_2_z.n[2] != 0)
        {// protects from dividing by zero.
            check_z_dims = ((process_topology_3.n[0]) % (process_topology_2_z.n[0]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_z.n[1]) == 0) 
            && (n[0] % (process_topology_2_z.nproc[0]) == 0) 
            && (n[0] % (process_topology_2_z.nproc[1]) == 0);
            
            if(self==0 && debug && !check_z_dims){
            fprintf(stderr,"Need to fix Z PENCILS z_procs(%d,%d,%d) 3d.ns(%d,%d,%d) 2d_z.ns(%d,%d,%d).... \n", 
                process_topology_2_z.nproc[0],
                process_topology_2_z.nproc[1],
                process_topology_2_z.nproc[2],
                process_topology_3.n[0],
                process_topology_3.n[1],
                process_topology_3.n[2],
                process_topology_2_z.n[0],
                process_topology_2_z.n[1],
                process_topology_2_z.n[2]);
            }
        
            //try swaping pencil dimensions if current setup pencil dimensions dont 
            //fit inside the cubes.
            if(!(check_z_dims) 
            && ((process_topology_3.n[0]) % (process_topology_2_z.n[1]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_z.n[0]) == 0))
            {

            if(self==0 && debug)
            fprintf(stderr,"Swaping Z pencils in initialization  (%d,%d,%d).... \n", 
                process_topology_2_z.nproc[0],
                process_topology_2_z.nproc[1],
                process_topology_2_z.nproc[2]);
            int temp=process_topology_2_z.nproc[0];
            process_topology_2_z.nproc[0] = process_topology_2_z.nproc[1];
            process_topology_2_z.nproc[1] = temp;
            process_topology_2_z.nproc[2] = process_topology_2_z.nproc[2];
            
            process_topology_2_z.n[0] = n[0] / process_topology_2_z.nproc[0];
            process_topology_2_z.n[1] = n[1] / process_topology_2_z.nproc[1];
            process_topology_2_z.n[2] = n[2] / process_topology_2_z.nproc[2];
            check_z_dims = ((process_topology_3.n[0]) % (process_topology_2_z.n[0]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_z.n[1]) == 0)
            && (n[0] % (process_topology_2_z.nproc[0]) == 0) 
            && (n[0] % (process_topology_2_z.nproc[1]) == 0);
            }
        } else {
            check_z_dims=false;
        }
        /*
            if that did not work, make a pencil that does if inside the 3d cuboids by 
            taking the cuboids dimensions (np1,np2,np3) and making pencils 
            (np1,np2*np3,1), or (np1*np3,np2,1) on the most evenly distributed 
            dimensions
        */
        if(!check_z_dims){
            if(self==0 && debug)
            fprintf(stderr,"MAKING Z PENCILS FIT zprocs(%d,%d,%d) z.ns(%d,%d,%d).... \n", 
                process_topology_2_z.nproc[0],
                process_topology_2_z.nproc[1],
                process_topology_2_z.nproc[2],
                process_topology_2_z.n[0],
                process_topology_2_z.n[1],
                process_topology_2_z.n[2]);
            
            process_topology_2_z.nproc[2]=1;
            if(process_topology_3.n[0]>process_topology_3.n[1])
            {
            process_topology_2_z.nproc[1]=process_topology_3.nproc[1]*process_topology_3.nproc[2];
            process_topology_2_z.nproc[0]=process_topology_3.nproc[0];
            if((n[0] % (process_topology_2_z.nproc[0]) != 0) 
            || (n[0] % (process_topology_2_z.nproc[1]) != 0))
            {
            process_topology_2_z.nproc[0]=process_topology_3.nproc[0]*process_topology_3.nproc[2];
            process_topology_2_z.nproc[1]=process_topology_3.nproc[1];
            }
            } else {
            process_topology_2_z.nproc[0]=process_topology_3.nproc[0]*process_topology_3.nproc[2];
            process_topology_2_z.nproc[1]=process_topology_3.nproc[1];
            if((n[0] % (process_topology_2_z.nproc[0]) != 0) 
            || (n[0] % (process_topology_2_z.nproc[1]) != 0))
            {
            process_topology_2_z.nproc[1]=process_topology_3.nproc[1]*process_topology_3.nproc[2];
            process_topology_2_z.nproc[0]=process_topology_3.nproc[0];
            }
            }
            process_topology_2_z.n[0] = n[0] / process_topology_2_z.nproc[0];
            process_topology_2_z.n[1] = n[1] / process_topology_2_z.nproc[1];
            process_topology_2_z.n[2] = n[2] / process_topology_2_z.nproc[2];
            if(self==0 && debug)
            fprintf(stderr,"MAKING Z PENCILS FIT AFTER zprocs(%d,%d,%d) z.ns(%d,%d,%d)...\n", 
                process_topology_2_z.nproc[0],
                process_topology_2_z.nproc[1],
                process_topology_2_z.nproc[2],
                process_topology_2_z.n[0],
                process_topology_2_z.n[1],
                process_topology_2_z.n[2]);
            if(process_topology_2_z.n[0] != 0 
            && process_topology_2_z.n[1] != 0 
            && process_topology_2_z.n[2] != 0)
            {// protects from dividing by zero.
            check_z_dims=((process_topology_3.n[0]) % (process_topology_2_z.n[0]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_z.n[1]) == 0)
            && (n[0] % (process_topology_2_z.nproc[0]) == 0) 
            && (n[0] % (process_topology_2_z.nproc[1]) == 0);
            } else {
            check_z_dims=false;
            }
        }
            
        if (debug && 0 == self) {
            fprintf(stderr, "  2d_z: ");
            for (int i = 0; i < ndim; ++i) {
            fprintf(stderr, "%d%s", 
                process_topology_2_z.nproc[i], 
                separator(i, ndim));
            }
            fprintf(stderr, "\n");
        } 
        if(!check_z_dims && debug && (self==0)){
            FILE * outfile;
            outfile= fopen("error.data","a");
            fprintf(outfile,"Z DIMS FAILS:(%d,%d,%d) (%d,%d,%d) \n",
                process_topology_2_z.nproc[0],
                process_topology_2_z.nproc[1],
                process_topology_2_z.nproc[2], 
                process_topology_3.nproc[0],
                process_topology_3.nproc[1],
                process_topology_3.nproc[2]);
        }
        assert(check_z_dims);
        /*
        if this happends, it is because the dimensions were chosen incorrectly. 
        Either to many processors for the number of points in one dimenison (could 
        not do at least 1 point per processor), or the methods above could 
        not make a distribution of pencils that fit in the cubiods, which would 
        happen if the user gave numbers that wouldent work (we require the number 
        of processors in each dimension of the cuboid must be modulo the number of 
        points in that dimension, otherwise, this error will happen).
        */
        MPI_Cart_create(comm, 
                ndim, 
                process_topology_2_z.nproc, 
                period, 
                0, 
                &process_topology_2_z.cart);
        //find the cartesian coord of the current rank (for the z_pencil)
        coord_z_pencils(self,process_topology_2_z.self);

        #ifdef PRINT_DISTRIBUTION
        if(self == 0) {
            printf("distribution 2z: [%d:%d:%d]\n",
            process_topology_2_z.nproc[0],
            process_topology_2_z.nproc[1],
            process_topology_2_z.nproc[2]);
            fflush(stdout);
        }
        #endif



        // set up process grid with 2d decomposition (x_PENCILs)
        process_topology_2_x.nproc[0] = 1; // don't distribute outer dimension
        process_topology_2_x.nproc[1] = 0;
        process_topology_2_x.nproc[2] = 0;
        period[0] = period[1] = period[2] = 1;
        MPI_Dims_create(nproc, ndim, process_topology_2_x.nproc);
        process_topology_2_x.n[0] = n[0] / process_topology_2_x.nproc[0];
        process_topology_2_x.n[1] = n[1] / process_topology_2_x.nproc[1];
        process_topology_2_x.n[2] = n[2] / process_topology_2_x.nproc[2];
        //variable used to ensure that pencils created fit inside the cuboids, 
        //if not the code will assert out.
        bool check_x_dims = false;
        if(process_topology_2_x.n[0] != 0 
            && process_topology_2_x.n[1] != 0 
            && process_topology_2_x.n[2] != 0)
        {// protects from dividing by zero.
            check_x_dims = ((process_topology_3.n[2]) % (process_topology_2_x.n[2]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_x.n[1]) == 0) 
            && (n[0] % (process_topology_2_x.nproc[2]) == 0) 
            && (n[0] % (process_topology_2_x.nproc[1]) == 0);
            if(self==0 && debug && !check_x_dims)
            fprintf(stderr,"Need to fix X PENCILS x_procs(%d,%d,%d) 3d.ns(%d,%d,%d) 2d_x.ns(%d,%d,%d)...\n", 
                process_topology_2_x.nproc[0],
                process_topology_2_x.nproc[1],
                process_topology_2_x.nproc[2],
                process_topology_3.n[0],
                process_topology_3.n[1],
                process_topology_3.n[2],
                process_topology_2_x.n[0],
                process_topology_2_x.n[1],
                process_topology_2_x.n[2]);

            //try swaping pencil dimensions if current setup does not have pencils 
            //that fit inside cubes.
            if(!(check_x_dims) 
            && ((process_topology_3.n[2]) % (process_topology_2_x.n[1]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_x.n[2]) == 0))
            {
            if(self==0 && debug)
            fprintf(stderr,"Swaping X pencils in initialization .... \n");
            process_topology_2_x.nproc[0] = process_topology_2_x.nproc[0];
            int temp = process_topology_2_x.nproc[1];
            process_topology_2_x.nproc[1] = process_topology_2_x.nproc[2];
            process_topology_2_x.nproc[2] = temp;
        
            process_topology_2_x.n[0] = n[0] / process_topology_2_x.nproc[0];
            process_topology_2_x.n[1] = n[1] / process_topology_2_x.nproc[1];
            process_topology_2_x.n[2] = n[2] / process_topology_2_x.nproc[2];
            check_x_dims = ((process_topology_3.n[2]) % (process_topology_2_x.n[2]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_x.n[1]) == 0)
            && (n[0] % (process_topology_2_x.nproc[2]) == 0) 
            && (n[0] % (process_topology_2_x.nproc[1]) == 0);
            } 
        } else{
            check_x_dims=false;
        }
        /*
            if that did not work, make a pencil that does by taking the cuboid 
            (np1,np2,np3) and making pencils of the form (1,np2*np1,np3) or 
            (1,np2*np1,np3) depending on the most even distribution it can.
        */
        if(!check_x_dims){
            if(self==0 && debug)
            fprintf(stderr,"MAKING X PENCILS FIT xprocs(%d,%d,%d) x.ns(%d,%d,%d)...\n",
                process_topology_2_x.nproc[0],
                process_topology_2_x.nproc[1],
                process_topology_2_x.nproc[2],
                process_topology_2_x.n[0],
                process_topology_2_x.n[1],
                process_topology_2_x.n[2]);

            process_topology_2_x.nproc[0] = 1;
            if(process_topology_3.nproc[2] > process_topology_3.nproc[1])
            {
            process_topology_2_x.nproc[1] = process_topology_3.nproc[1]*process_topology_3.nproc[0];
            process_topology_2_x.nproc[2] = process_topology_3.nproc[2];
            if((n[0] % (process_topology_2_x.nproc[2]) != 0) 
            || (n[0] % (process_topology_2_x.nproc[0]) != 0))
            {
            process_topology_2_x.nproc[2]=process_topology_3.nproc[2]*process_topology_3.nproc[0];
            process_topology_2_x.nproc[1]=process_topology_3.nproc[1];
            }

            } else {
            process_topology_2_x.nproc[2] = process_topology_3.nproc[2]*process_topology_3.nproc[0];
            process_topology_2_x.nproc[1] = process_topology_3.nproc[1];
            if((n[0] % (process_topology_2_x.nproc[2]) != 0) 
            || (n[0] % (process_topology_2_x.nproc[0]) != 0))
            {
            process_topology_2_x.nproc[1]=process_topology_3.nproc[1]*process_topology_3.nproc[0];
            process_topology_2_x.nproc[2]=process_topology_3.nproc[2];
            }
            }
            process_topology_2_x.n[0] = n[0] / process_topology_2_x.nproc[0];
            process_topology_2_x.n[1] = n[1] / process_topology_2_x.nproc[1];
            process_topology_2_x.n[2] = n[2] / process_topology_2_x.nproc[2];
            if(self==0 && debug)
            fprintf(stderr,"MAKING X PENCILS FIT AFTER xprocs(%d,%d,%d) x.ns(%d,%d,%d)...\n",
                process_topology_2_x.nproc[0],
                process_topology_2_x.nproc[1],
                process_topology_2_x.nproc[2],
                process_topology_2_x.n[0],
                process_topology_2_x.n[1],
                process_topology_2_x.n[2]);
            if(process_topology_2_x.n[0] != 0 
            && process_topology_2_x.n[1] != 0 
            && process_topology_2_x.n[2] != 0)
            {// protects from dividing by zero.
            check_x_dims = ((process_topology_3.n[2]) % (process_topology_2_x.n[2]) == 0) 
            && ((process_topology_3.n[1]) % (process_topology_2_x.n[1]) == 0)
            && (n[0] % (process_topology_2_x.nproc[2]) == 0) 
            && (n[0] % (process_topology_2_x.nproc[1]) == 0);
            } else {
            check_x_dims=false;
            }  
        }
        
        if (debug && 0 == self) {
            fprintf(stderr, "  2d_x: ");
            for (int i = 0; i < ndim; ++i) {
            fprintf(stderr, "%d%s", 
                process_topology_2_x.nproc[i], 
                separator(i, ndim));
            }
            fprintf(stderr, "\n");
        }
        if(!check_x_dims && debug && (self==0)){
            FILE * outfile;
            outfile= fopen("error.data","a");
            fprintf(outfile,"X DIMS FAILS:(%d,%d,%d) (%d,%d,%d) \n",
                process_topology_2_x.nproc[0],
                process_topology_2_x.nproc[1],
                process_topology_2_x.nproc[2], 
                process_topology_3.nproc[0],
                process_topology_3.nproc[1],
                process_topology_3.nproc[2]);
        }
        assert(check_x_dims);
        /*
        if this happends, it is because the dimensions were chosen incorrectly. 
        Either to many processors for the number of points in one dimenison (could 
        not do at least 1 point per processor), or the methods above could not make 
        a distribution of pencils that fit in the cubiods, which would happen if the 
        user gave numbers that wouldent work (we require the number of processors in 
        each dimension of the cuboid must be modulo the number of points in that 
        dimension, otherwise, this error will happen).
        */
        MPI_Cart_create(comm, 
                ndim, 
                process_topology_2_x.nproc, 
                period, 
                0, 
                &process_topology_2_x.cart);
        coord_x_pencils(self, process_topology_2_x.self);

        #ifdef PRINT_DISTRIBUTION
        if(self == 0) {
            printf("distribution 2x: [%d:%d:%d]\n",
            process_topology_2_x.nproc[0],
            process_topology_2_x.nproc[1],
            process_topology_2_x.nproc[2]);
            fflush(stdout);
        }
        #endif
        


        // set up process grid with 2d decomposition (y_PENCILs)
        process_topology_2_y.nproc[0] = 0;
        process_topology_2_y.nproc[1] = 1; // don't distribute outer dimension
        process_topology_2_y.nproc[2] = 0;
        period[0] = period[1] = period[2] = 1;
        MPI_Dims_create(nproc, ndim, process_topology_2_y.nproc);
        process_topology_2_y.n[0] = n[0] / process_topology_2_y.nproc[0];
        process_topology_2_y.n[1] = n[1] / process_topology_2_y.nproc[1];
        process_topology_2_y.n[2] = n[2] / process_topology_2_y.nproc[2];
        //variable used to ensure that pencils created fit inside the cuboids, 
        //if not the code will assert out.
        bool check_y_dims=false;
        if(process_topology_2_y.n[0] != 0 
            && process_topology_2_y.n[1] != 0 
            && process_topology_2_y.n[2] != 0)
        {// protects from dividing by zero.
            check_y_dims = (((process_topology_3.n[2]) % (process_topology_2_y.n[2]) == 0) 
                    && ((process_topology_3.n[0]) % (process_topology_2_y.n[0]) == 0) 
                    && (n[0] % (process_topology_2_y.nproc[2]) == 0) 
                    && (n[0] % (process_topology_2_y.nproc[0]) == 0));
            if(self==0 && debug && !check_y_dims)
            fprintf(stderr,"Need to fix Y PENCILS y_procs(%d,%d,%d) 3d.ns(%d,%d,%d) 2d_y.ns(%d,%d,%d)...\n",
                process_topology_2_y.nproc[0],
                process_topology_2_y.nproc[1],
                process_topology_2_y.nproc[2],
                process_topology_3.n[0],
                process_topology_3.n[1],
                process_topology_3.n[2],
                process_topology_2_y.n[0],
                process_topology_2_y.n[1],
                process_topology_2_y.n[2]);
            //try swaping pencil dimensions if the current dimension of the pencils 
            //does not fit inside the cubes.
            if(!(check_y_dims) 
            && ((process_topology_3.n[2]) % (process_topology_2_y.n[0]) == 0) 
            && ((process_topology_3.n[0]) % (process_topology_2_y.n[2]) == 0))
            {
            if(self==0 && debug)
            fprintf(stderr,"Swaping Y pencils in initialization .... \n");
            
            int temp = process_topology_2_y.nproc[0];
            process_topology_2_y.nproc[0] = process_topology_2_y.nproc[2];
            process_topology_2_y.nproc[2] = temp;
            process_topology_2_y.nproc[1] = process_topology_2_y.nproc[1];
            
            process_topology_2_y.n[0] = n[0] / process_topology_2_y.nproc[0];
            process_topology_2_y.n[1] = n[1] / process_topology_2_y.nproc[1];
            process_topology_2_y.n[2] = n[2] / process_topology_2_y.nproc[2];
            check_y_dims = (((process_topology_3.n[2]) % (process_topology_2_y.n[2]) == 0) 
                    && ((process_topology_3.n[0]) % (process_topology_2_y.n[0]) == 0) 
                    && (n[0] % (process_topology_2_y.nproc[2]) == 0) 
                    && (n[0] % (process_topology_2_y.nproc[0]) == 0));
            }
        } else {
            check_y_dims = false;
        }
        /*
        if that did not work, make a pencil that does by taking the cuboid 
        (np1,np2,np3) and making pencils of the form (np1,1,np3*np2) or 
        (np1*np2,1,np3) depending on the most even distribution it can.
        */
        if(!check_y_dims){
            if(self==0 && debug)
            fprintf(stderr,"MAKING Y PENCILS FIT yprocs(%d,%d,%d) y.ns(%d,%d,%d)...\n", 
                process_topology_2_y.nproc[0],
                process_topology_2_y.nproc[1],
                process_topology_2_y.nproc[2],
                process_topology_2_y.n[0],
                process_topology_2_y.n[1],
                process_topology_2_y.n[2]);
            
            process_topology_2_y.nproc[1]=1;
            if(process_topology_3.nproc[2] > process_topology_3.nproc[0])
            {
            process_topology_2_y.nproc[0] = process_topology_3.nproc[0]*process_topology_3.nproc[1];
            process_topology_2_y.nproc[2] = process_topology_3.nproc[2];
            if((n[0] % (process_topology_2_y.nproc[2]) != 0) 
            || (n[0] % (process_topology_2_y.nproc[0]) != 0))
            {
            process_topology_2_y.nproc[2] = process_topology_3.nproc[2]*process_topology_3.nproc[1];
            process_topology_2_y.nproc[0] = process_topology_3.nproc[0];
            }
            } else {
            process_topology_2_y.nproc[2] = process_topology_3.nproc[2]*process_topology_3.nproc[1];
            process_topology_2_y.nproc[0] = process_topology_3.nproc[0];
            if((n[0] % (process_topology_2_y.nproc[2]) != 0) 
            || (n[0] % (process_topology_2_y.nproc[0]) != 0))
            {
            process_topology_2_y.nproc[0] = process_topology_3.nproc[0]*process_topology_3.nproc[1];
            process_topology_2_y.nproc[2] = process_topology_3.nproc[2];
            }
            }
            
            process_topology_2_y.n[0] = n[0] / process_topology_2_y.nproc[0];
            process_topology_2_y.n[1] = n[1] / process_topology_2_y.nproc[1];
            process_topology_2_y.n[2] = n[2] / process_topology_2_y.nproc[2];
            if(self==0 && debug)
            fprintf(stderr,"MAKING Y PENCILS FIT AFTER yprocs(%d,%d,%d) y.ns(%d,%d,%d)...\n",
                process_topology_2_y.nproc[0],
                process_topology_2_y.nproc[1],
                process_topology_2_y.nproc[2],
                process_topology_2_y.n[0],
                process_topology_2_y.n[1],
                process_topology_2_y.n[2]);
            if(process_topology_2_y.n[0] != 0 && process_topology_2_y.n[1] != 0 
            && process_topology_2_y.n[2] != 0)
            {// protects from dividing by zero.
            check_y_dims = (((process_topology_3.n[2]) % (process_topology_2_y.n[2]) == 0) 
                    && ((process_topology_3.n[0]) % (process_topology_2_y.n[0]) == 0) 
                    && (n[0] % (process_topology_2_y.nproc[2]) == 0) 
                    && (n[0] % (process_topology_2_y.nproc[0]) == 0));
            } else {
            check_y_dims=false;
            }
        }
        
        if (debug && 0 == self) {
            fprintf(stderr, "  2d_y: ");
            for (int i = 0; i < ndim; ++i) {
            fprintf(stderr, "%d%s", 
                process_topology_2_y.nproc[i], 
                separator(i, ndim));
            }
            fprintf(stderr, "\n");
        }
        if(!check_y_dims && debug && (self==0)){
            FILE * outfile;
            outfile = fopen("error.data","a");
            fprintf(outfile,"Y DIMS FAILS:(%d,%d,%d) (%d,%d,%d) \n",
                process_topology_2_y.nproc[0],
                process_topology_2_y.nproc[1],
                process_topology_2_y.nproc[2], 
                process_topology_3.nproc[0],
                process_topology_3.nproc[1],
                process_topology_3.nproc[2]);
        }
        assert(check_y_dims);
        /*
        if this happends, it is because the dimensions were chosen incorrectly. 
        Either to many processors for the number of points in one dimenison (could 
        not do at least 1 point per processor), or the methods above could 
        not make a distribution of pencils that fit in the cubiods, which would 
        happen if the user gave numbers that wouldent work (we require the number of 
        processors in each dimension of the cuboid must be modulo the number of 
        points in that dimension, otherwise, this error will happen).
        */
        MPI_Cart_create(comm, 
                ndim, 
                process_topology_2_y.nproc, 
                period, 
                0, 
                &process_topology_2_y.cart);
        //find the cartesian coord of the current rank (for the y_pencil)
        coord_y_pencils(self,process_topology_2_y.self);

        #ifdef PRINT_DISTRIBUTION
        if(self == 0) {
            printf("distribution 2y: [%d:%d:%d]\n",
            process_topology_2_y.nproc[0],
            process_topology_2_y.nproc[1],
            process_topology_2_y.nproc[2]);
            fflush(stdout);
        }
        #endif


        
        if (debug) {
            int myrank_cube;
            rank_cube(&myrank_cube,process_topology_3.self);
            int myrank_x;
            rank_x_pencils(&myrank_x,process_topology_2_x.self);
            int myrank_y;
            rank_y_pencils(&myrank_y,process_topology_2_y.self);
            int myrank_z;
            rank_z_pencils(&myrank_z,process_topology_2_z.self);
            if(myrank_z != self 
            || myrank_y != self 
            || myrank_x != self 
            || myrank_cube != self)
            abort(); //means ranks were calculated wrong.
            if (0 == self) {
            fprintf(stderr, "Process map:\n");
            }
            for (int p = 0; p < nproc; ++p) {
            MPI_Barrier(comm);
            if (p == self) {
            fprintf(stderr, "  %d: 1d = (%d, %d, %d), 2d_x = (%d, %d, %d) rank is= %d,2d_y = (%d, %d, %d) rank is= %d,2d_z = (%d, %d, %d) rank is= %d, 3d = (%d, %d, %d). rank is= %d\n",
                self,
                process_topology_1.self[0], 
                process_topology_1.self[1], 
                process_topology_1.self[2],
                process_topology_2_x.self[0], 
                process_topology_2_x.self[1], 
                process_topology_2_x.self[2],
                myrank_x,
                process_topology_2_y.self[0], 
                process_topology_2_y.self[1], 
                process_topology_2_y.self[2],
                myrank_y,
                process_topology_2_z.self[0], 
                process_topology_2_z.self[1], 
                process_topology_2_z.self[2],
                myrank_z,
                process_topology_3.self[0], 
                process_topology_3.self[1], 
                process_topology_3.self[2],
                myrank_cube);
            }
            }
        }

        //allocate size of buffers used to hold pencil chunks of data in the 
        //distribution routines for 3d to 1d and vica versa.
        int buff_z_chunk = process_topology_2_z.n[0]*process_topology_2_z.n[1]*process_topology_3.n[2];
        int buff_y_chunk = process_topology_2_y.n[0]*process_topology_2_y.n[2]*process_topology_3.n[1];
        int buff_x_chunk = process_topology_2_x.n[1]*process_topology_2_x.n[2]*process_topology_3.n[0];
        int buff_size = 0;
        if(buff_z_chunk > buff_y_chunk){
            buff_size=buff_z_chunk;
        } else {
            buff_size=buff_y_chunk;
        }
        if(buff_x_chunk > buff_size)
            buff_size = buff_x_chunk;
        
        d2_chunk=(T *) malloc(sizeof(T)*buff_size);
        d3_chunk=(T *) malloc(sizeof(T)*buff_size);

    }

    template<class T, class MPI_T>
    distribution_t<T,MPI_T>::~distribution_t(){
        MPI_Comm_free(&process_topology_1.cart);
        MPI_Comm_free(&process_topology_2_x.cart);
        MPI_Comm_free(&process_topology_2_y.cart);
        MPI_Comm_free(&process_topology_2_z.cart);
        MPI_Comm_free(&process_topology_3.cart);
        free(d2_chunk);
        free(d3_chunk);
    }
    
    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::assert_commensurate(){
        for (int i = 0; i < 3; ++i) {
        #if defined(PENCIL)
            assert(0 == (n[i] % process_topology_2_x.nproc[i]));
            assert(0 == (n[i] % process_topology_2_y.nproc[i]));
            assert(0 == (n[i] % process_topology_2_z.nproc[i]));
        #else
            assert(0 == (n[i] % process_topology_1.nproc[i]));
        #endif
            assert(0 == (n[i] % process_topology_3.nproc[i]));
        }

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::dist_1_to_3(const T* a, T* b){
        if (USE_SLAB_WORKAROUND){
            redistribute_slab(a,b,REDISTRIBUTE_1_TO_3);
        } else {
            redistribute(a,b,REDISTRIBUTE_1_TO_3);
        }
    }

    
    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::dist_3_to_1(const T* a, T* b){
        if (USE_SLAB_WORKAROUND){
            redistribute_slab(a,b,REDISTRIBUTE_3_TO_1);
        } else {
            redistribute(a,b,REDISTRIBUTE_3_TO_1);
        }
    }

    template<class T>
    void make_mpi_subarray(int ndims, const int* array_of_sizes, const int* array_of_subsizes, const int* array_of_starts, MPI_Datatype* new_dt);

    template<>
    void make_mpi_subarray<complexDoubleHost>(int ndims, const int* array_of_sizes, const int* array_of_subsizes, const int* array_of_starts, MPI_Datatype* new_dt){
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE_COMPLEX, new_dt);
        MPI_Type_commit(new_dt);
    }

    template<>
    void make_mpi_subarray<complexFloatHost>(int ndims, const int* array_of_sizes, const int* array_of_subsizes, const int* array_of_starts, MPI_Datatype* new_dt){
        MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_COMPLEX, new_dt);
        MPI_Type_commit(new_dt);
    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::redistribute(const T* a, T* b, redist_t direction){
        int remaining_dim[3];
        MPI_Comm subgrid_cart;
        int subgrid_self;
        int subgrid_nproc;

        // exchange data with processes in a 2-d slab of 3-d subdomains

        remaining_dim[0] = 0;
        remaining_dim[1] = 1;
        remaining_dim[2] = 1;
        MPI_Cart_sub(process_topology_3.cart, remaining_dim, &subgrid_cart);
        MPI_Comm_rank(subgrid_cart, &subgrid_self);
        MPI_Comm_size(subgrid_cart, &subgrid_nproc);

        for (int p = 0; p < subgrid_nproc; ++p) {
        int d1_peer = (subgrid_self + p) % subgrid_nproc;
        int d3_peer = (subgrid_self - p + subgrid_nproc) % subgrid_nproc;
        int coord[2];
        int sizes[3];
        int subsizes[3];
        int starts[3];
        MPI_Datatype d1_type;
        MPI_Datatype d3_type;

        MPI_Cart_coords(subgrid_cart, d1_peer, 2, coord);
        if (0) {
            int self;
            MPI_Comm_rank(MPI_COMM_WORLD, &self);
            fprintf(stderr, "%d: d1_peer, d1_coord, d3_peer = %d, (%d, %d), %d\n",
                self, d1_peer, coord[0], coord[1], d3_peer);
        }

        // create dataypes representing a subarray in the 1- and 3-d distributions

        sizes[0] = process_topology_1.n[0];
        sizes[1] = process_topology_1.n[1];
        sizes[2] = process_topology_1.n[2];
        subsizes[0] = process_topology_1.n[0];
        subsizes[1] = process_topology_3.n[1];
        subsizes[2] = process_topology_3.n[2];
        starts[0] = 0;
        starts[1] = coord[0] * process_topology_3.n[1];
        starts[2] = coord[1] * process_topology_3.n[2];
        make_mpi_subarray<T>(3,sizes,subsizes,starts,&d1_type);
        //MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE_COMPLEX, &d1_type);
        //MPI_Type_commit(&d1_type);

        sizes[0] = process_topology_3.n[0];
        sizes[1] = process_topology_3.n[1];
        sizes[2] = process_topology_3.n[2];
        subsizes[0] = process_topology_1.n[0];
        subsizes[1] = process_topology_3.n[1];
        subsizes[2] = process_topology_3.n[2];
        starts[0] = d3_peer * process_topology_1.n[0];
        starts[1] = 0;
        starts[2] = 0;
        make_mpi_subarray<T>(3,sizes,subsizes,starts,&d3_type);
        //MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE_COMPLEX, &d3_type);
        //MPI_Type_commit(&d3_type);

        // exchange data

        if (direction == REDISTRIBUTE_3_TO_1) {
            MPI_Sendrecv((char *) a, 1, d3_type, d3_peer, 0,
                (char *) b, 1, d1_type, d1_peer, 0,
                subgrid_cart, MPI_STATUS_IGNORE);
        } else if (direction == REDISTRIBUTE_1_TO_3) {
            MPI_Sendrecv((char *) a, 1, d1_type, d1_peer, 0,
                (char *) b, 1, d3_type, d3_peer, 0,
                subgrid_cart, MPI_STATUS_IGNORE);
        } else {
            abort();
        }

        // free datatypes

        MPI_Type_free(&d1_type);
        MPI_Type_free(&d3_type);
        }

        MPI_Comm_free(&subgrid_cart);

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::redistribute_slab(const T* a, T* b, redist_t direction){
        int remaining_dim[3];
        MPI_Comm subgrid_cart;
        int subgrid_self;
        int subgrid_nproc;
        ptrdiff_t d1_slice = process_topology_1.n[1] * process_topology_1.n[2] * sizeof(T);
        ptrdiff_t d3_slice = process_topology_3.n[1] * process_topology_3.n[2] * sizeof(T);

        // exchange data with processes in a 2-d slab of 3-d subdomains

        remaining_dim[0] = 0;
        remaining_dim[1] = 1;
        remaining_dim[2] = 1;
        MPI_Cart_sub(process_topology_3.cart, remaining_dim, &subgrid_cart);
        MPI_Comm_rank(subgrid_cart, &subgrid_self);
        MPI_Comm_size(subgrid_cart, &subgrid_nproc);

        for (int p = 0; p < subgrid_nproc; ++p) {
        int coord[2];
        int d1_peer = (subgrid_self + p) % subgrid_nproc;
        int d3_peer = (subgrid_self - p + subgrid_nproc) % subgrid_nproc;

        MPI_Cart_coords(subgrid_cart, d1_peer, 2, coord);
        if (0) {
            int self;
            MPI_Comm_rank(MPI_COMM_WORLD, &self);
            fprintf(stderr, "%d: d1_peer, d1_coord, d3_peer = %d, (%d, %d), %d\n",
                self, d1_peer, coord[0], coord[1], d3_peer);
        }

        for (int slice = 0; slice < process_topology_1.n[0]; ++slice) {
            int sizes[2];
            int subsizes[2];
            int starts[2];
            MPI_Datatype d1_type;
            MPI_Datatype d3_type;
            ptrdiff_t d1_offset = slice * d1_slice;
            ptrdiff_t d3_offset = (slice + d3_peer * process_topology_1.n[0]) * d3_slice;
            
            // create subarray dataypes representing the slice subarray in the 1- and 3-d distributions
            
            sizes[0] = process_topology_1.n[1];
            sizes[1] = process_topology_1.n[2];
            subsizes[0] = process_topology_3.n[1];
            subsizes[1] = process_topology_3.n[2];
            starts[0] = coord[0] * process_topology_3.n[1];
            starts[1] = coord[1] * process_topology_3.n[2];
            make_mpi_subarray<T>(2,sizes,subsizes,starts,&d1_type);
            //MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE_COMPLEX, &d1_type);
            //MPI_Type_commit(&d1_type);
            
            /*MPI_Type_contiguous(process_topology_3.n[1] * process_topology_3.n[2],
                    MPI_DOUBLE_COMPLEX,
                    &d3_type);*/
            MPI_Type_contiguous(process_topology_3.n[1] * process_topology_3.n[2] * sizeof(T),
                    MPI_BYTE,
                    &d3_type);
            MPI_Type_commit(&d3_type);
            
            // exchange data
            
            if (direction == REDISTRIBUTE_3_TO_1) {
        MPI_Sendrecv((char *) a + d3_offset, 1, d3_type, d3_peer, 0,
                    (char *) b + d1_offset, 1, d1_type, d1_peer, 0,
                    subgrid_cart, MPI_STATUS_IGNORE);
            } else if (direction == REDISTRIBUTE_1_TO_3) {
        MPI_Sendrecv((char *) a + d1_offset, 1, d1_type, d1_peer, 0,
                    (char *) b + d3_offset, 1, d3_type, d3_peer, 0,
                    subgrid_cart, MPI_STATUS_IGNORE);
            } else {
        abort();
            }
            
            // free datatypes
            
            MPI_Type_free(&d1_type);
            MPI_Type_free(&d3_type);
        }
        }

        MPI_Comm_free(&subgrid_cart);

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::dist_2_to_3(const T* a, T* b, int z_dim){
        redistribute_2_and_3(a,b,REDISTRIBUTE_2_TO_3,z_dim);
    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::dist_3_to_2(const T* a, T* b, int z_dim){
        redistribute_2_and_3(a,b,REDISTRIBUTE_3_TO_2,z_dim);
    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::redistribute_2_and_3(const T* a, T* b, redist_t direction, int z_dim){
        int self = process_topology_1.self[0];
        int npeers;
        int me=0;//determines which processor to print
        bool print_me=false; //prints info on proccessor whose rank = me.
        bool print_mess=false;//prints communication sends and recieves without actually doing the comms(intended to debug comm hangs).
        bool print_result=false /*true*/;//prints a line in a file called "passed.data" which happends if the code runs completly.
        assert(z_dim==0||z_dim==1||z_dim==2);
        int x_dim=0,y_dim=0;
        //x_dim, y_dim and z_dim are the dimensions of the x,y,z axis of the pencil with respect to the original axis(where index 2 is into the grid, 1 is vertical translation and 0 is horizontal).
        switch(z_dim){
            case 0: x_dim=1; y_dim=2; 
                if((self == me) && print_me)fprintf(stderr, "DOING X PENCILS!...\n"); break;
            case 1: x_dim=2; y_dim=0;
                if((self == me && print_me))fprintf(stderr, "DOING Y PENCILS!...\n"); break;
            case 2: x_dim=0; y_dim=1;
                if((self == me && print_me))fprintf(stderr, "DOING Z PENCILS!...\n"); break;
            default: assert("incorrect inputed dimension");
        }
        
        // assuming dimensions are all commensurate, then the number of
        // peers to exchange with is the number of processes in the z_dimension
        // direction in the 3d distribution
        npeers = process_topology_3.nproc[z_dim]; //picked last direction (lets say into the grid)
        
        // book-keeping for the processor translation in the x-y plane
        int p0 = 0;
        int p1 = 0;
        int p1max = 0;
        
        MPI_Request req1=MPI_REQUEST_NULL;
        MPI_Request req2=MPI_REQUEST_NULL;
        
        int pencil_sizes[3];
        int cube_sizes[3];
        int subsizes[3];
        
        
        cube_sizes[x_dim] = process_topology_3.n[x_dim];
        cube_sizes[y_dim] = process_topology_3.n[y_dim]; 
        cube_sizes[z_dim] = process_topology_3.n[z_dim];
        
        //set varibles used to calculate the subarrays of each pencil and cube.
        switch(z_dim){
            case 0: 
            p1max = process_topology_2_x.nproc[x_dim] / process_topology_3.nproc[x_dim] - 1; 
            //find out the size of the chunk you need to use (stored in subsizes), and set sizes to the local size of the pencil.
            //The x and y dimensions of the subchunck will be the dimensions of the pencil (since the code asserts at the beginning that all pencils fit inside the 3d cuboid.)
            //The z dimension will be the dimension of the cuboid, since this will always be <= to the z_dim of the pencil.
            pencil_sizes[x_dim] = process_topology_2_x.n[x_dim];
            pencil_sizes[y_dim] = process_topology_2_x.n[y_dim];  
            pencil_sizes[z_dim] = process_topology_2_x.n[z_dim]; 
            subsizes[x_dim] = process_topology_2_x.n[x_dim];
            subsizes[y_dim] = process_topology_2_x.n[y_dim];   
            break;
            case 1: 
            p1max = process_topology_2_y.nproc[x_dim] / process_topology_3.nproc[x_dim] - 1; 
            pencil_sizes[x_dim] = process_topology_2_y.n[x_dim];
            pencil_sizes[y_dim] = process_topology_2_y.n[y_dim];  
            pencil_sizes[z_dim] = process_topology_2_y.n[z_dim]; 
            subsizes[x_dim] = process_topology_2_y.n[x_dim];
            subsizes[y_dim] = process_topology_2_y.n[y_dim];   
            break;
            case 2: 
            p1max = process_topology_2_z.nproc[y_dim] / process_topology_3.nproc[y_dim] - 1; 
            pencil_sizes[x_dim] = process_topology_2_z.n[x_dim];
            pencil_sizes[y_dim] = process_topology_2_z.n[y_dim];  
            pencil_sizes[z_dim] = process_topology_2_z.n[z_dim]; 
            subsizes[x_dim] = process_topology_2_z.n[x_dim];
            subsizes[y_dim] = process_topology_2_z.n[y_dim];   
            break;
        }
        subsizes[z_dim] = process_topology_3.n[z_dim];
        int chunk_size=subsizes[0]*subsizes[1]*subsizes[2];//size of data chunks that will be communicated between pencil and cube distributions.
        
        //set variables that will be used to find pencils chunks
        int pencil_dims[3]={0,0,0};// size of entire pencil in its local coord system 
        int local_sizes[3]={0,0,0}; //size of chunck in its local coord system.
        if(z_dim==2){
            local_sizes[0]=subsizes[0];
            local_sizes[1]=subsizes[1];
            local_sizes[2]=subsizes[2];
            pencil_dims[0]=process_topology_2_z.n[0];//pencil dims in grid coord system (where index 2 is the z direction).
            pencil_dims[1]=process_topology_2_z.n[1];
            pencil_dims[2]=process_topology_2_z.n[2];
        }
        else if(z_dim==1){
            
            local_sizes[0]=subsizes[0];
            local_sizes[1]=subsizes[2];
            local_sizes[2]=subsizes[1];
            pencil_dims[0]=process_topology_2_y.n[0];
            pencil_dims[1]=process_topology_2_y.n[2];
            pencil_dims[2]=process_topology_2_y.n[1];
        }
        else if(z_dim==0){
            local_sizes[0]=subsizes[2];
            local_sizes[1]=subsizes[1];
            local_sizes[2]=subsizes[0];
            pencil_dims[0]=process_topology_2_x.n[2];
            pencil_dims[1]=process_topology_2_x.n[1];
            pencil_dims[2]=process_topology_2_x.n[0];
        }
        
        if((self == me) && print_me)fprintf(stderr, "%d, %d, %d, %d Dimensions!...\n", x_dim,y_dim,z_dim, p1max);
        
        // communicate with our peers
        for (int p = 0; p < npeers; ++p) {
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d Made it beg-for!...\n", self,p, npeers);
            
            int d2_coord[3];
            int d2_peer;
            int d2_peer_coord[3];
            int d3_coord[3];
            int d3_peer;
            int d3_peer_coord[3];
            int recv_peer;
            int send_peer;
            int d2_array_start[3];
            int d3_array_start[3];
            //turn the processor coordinate into one specified by the number of data points in each dimension.
            for (int i = 0; i < 3; ++i) {
            switch(z_dim){
            case 0: d2_coord[i]  = process_topology_2_x.self[i] * process_topology_2_x.n[i]; break;
            case 1: d2_coord[i]  = process_topology_2_y.self[i] * process_topology_2_y.n[i]; break;
            case 2: d2_coord[i]  = process_topology_2_z.self[i] * process_topology_2_z.n[i]; break;
            }
            }
            //over every iteration of the loop, transverse down the pencil (since it will be divided in chunks whose coordinates will only differ in the z_dimension.
            d2_coord[z_dim] += p * process_topology_3.n[z_dim]; 
            
            
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d Coord!...\n", d2_coord[0],d2_coord[1],d2_coord[2]);
            
            
            //d2_array_start is the starting index of the chunk in the pencils local coordinates.
            d2_array_start[0] = d2_coord[x_dim] % pencil_sizes[x_dim]; 
            d2_array_start[1] = d2_coord[y_dim] % pencil_sizes[y_dim]; 
            d2_array_start[2] = d2_coord[z_dim] % pencil_sizes[z_dim]; 
            
            if (DEBUG_CONDITION || ((self== me) && print_me)) {
            fprintf(stderr,
                "%d: pencil_sizes=(%d,%d,%d), cube_sizes=(%d,%d,%d), subsizes=(%d,%d,%d),d2_coord=(%d,%d,%d), d2_array_start=(%d,%d,%d) \n",
                self,
                pencil_sizes[0], pencil_sizes[1], pencil_sizes[2],
                cube_sizes[0], cube_sizes[1], cube_sizes[2],
                subsizes[0], subsizes[1], subsizes[2],
                d2_coord[0], d2_coord[1], d2_coord[2],
                d2_array_start[0],d2_array_start[1],d2_array_start[2]);
            }
            
            
            //if making cuboids from pencils, right here we need to fill the d2_chunk array with the data that later needs to be sent to a cuboid.
                //The array is a chunk of the pencil and is why we needed to calculate the starting index for the array in the local coordinates.
            if(direction == REDISTRIBUTE_2_TO_3){	
            int64_t ch_indx=0;
            int dims_size=pencil_dims[0]*pencil_dims[1]*pencil_dims[2];
            
            #ifdef DIST_OMP
            int last_i = d2_array_start[0] - 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif

            for(int i0=d2_array_start[0];i0<d2_array_start[0]+local_sizes[0];i0++){
            
            #ifdef DIST_OMP
            if(last_i != i0 - 1) {
                ch_indx = 0;
                for(int i=d2_array_start[0]; i<i0; i++)
                for(int i1=d2_array_start[1];i1<d2_array_start[1]+local_sizes[1];i1++)
                    for(int i2=d2_array_start[2];i2<d2_array_start[2]+local_sizes[2];i2++)
                ch_indx++;
            }
            last_i = i0;
            #endif
            
            for(int i1=d2_array_start[1];i1<d2_array_start[1]+local_sizes[1];i1++){
            for(int i2=d2_array_start[2];i2<d2_array_start[2]+local_sizes[2];i2++){
                int64_t local_indx=pencil_dims[2]*(pencil_dims[1]*i0+i1) + i2;
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                d2_chunk[ch_indx]=a[local_indx];
                ch_indx++;
            }
            }
            }
            
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d, pencil_dims!...\n", pencil_dims[0],pencil_dims[1],pencil_dims[2]);
            }
            
            // what peer in the 3d distribution owns this subarray? 
            for (int i = 0; i < 3; ++i) {
            d3_peer_coord[i] = d2_coord[i] / process_topology_3.n[i];
            }
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d Cube that hits pencil coord!...\n",d3_peer_coord[0],d3_peer_coord[1],d3_peer_coord[2]);
            //find the rank of this peer.
            switch(z_dim){
            case 0: MPI_Cart_rank(process_topology_3.cart, d3_peer_coord, &d3_peer); break;
            case 1: MPI_Cart_rank(process_topology_3.cart, d3_peer_coord, &d3_peer); break;
            case 2: MPI_Cart_rank(process_topology_3.cart, d3_peer_coord, &d3_peer); break;
            }
            if((self == me) && print_me)fprintf(stderr, "%d, %d, Made it half way!...\n", self,p);
            if((self == me) && print_me)fprintf(stderr, "%d, %d, PEER!...\n", self,d3_peer);
            
            //By here in the for loop, we have broken the pencil into a chunk and found which cuboid it resides; over every iteration, the for-loop will break up the pencil in the z_dimension.
            //From here on we do the opposite. We divide the cuboid into chunks (that are the same size as the ones in the pencil), and determine which pencils own these chunks.
            
            
            // what is the coordinate of my pth subarray in the 3d distribution?
            for (int i = 0; i < 3; ++i) {
            switch(z_dim){
            case 0: d3_coord[i]  = process_topology_3.self[i] * process_topology_3.n[i]; break;
            case 1: d3_coord[i]  = process_topology_3.self[i] * process_topology_3.n[i]; break;
            case 2: d3_coord[i]  = process_topology_3.self[i] * process_topology_3.n[i]; break;
            }
            }
            
            //now unlike above, we dont need to iterate in the z_dim, because for each processor its subarrays inward dimension is already set by the cubes z_dim.
            //Instead, each iteration of the for-loop will look at different subarrays whose locations in the cuboid differ by local x and y coords.
            
            switch(z_dim){
            //p1 is a place holder for the first translation . The outside for-loop will increment the coord in that direction, say x_dim, 
            //and keep doing so until all of the chunks in that dimension are calculated. Then it will increment p0 in the other dimension (in this example the y) 
            //and repeat until all of the subchunks in the x and y dimensions are calculated.
            //are found. 
            //Note: p0 and p1 will increment different dimensions depending of whether it is using the x y or z pencils, this is because the set up of the coordinate system for each 
            //pencil is different and to ensure that no communications hang up later, the directions coded below are unique for each type of pencil.
            case 0:
            d3_coord[y_dim] += p0 * process_topology_2_x.n[y_dim]; 
            d3_coord[x_dim] += p1 * process_topology_2_x.n[x_dim]; 
            break;
            case 1:
            d3_coord[y_dim] += p0 * process_topology_2_y.n[y_dim]; 
            d3_coord[x_dim] += p1 * process_topology_2_y.n[x_dim]; 
            break;
            case 2:
            d3_coord[x_dim] += p0 * process_topology_2_z.n[x_dim]; 
            d3_coord[y_dim] += p1 * process_topology_2_z.n[y_dim]; 
            break;
            }
            if (p1 == p1max) {
            p0++;
            p1 = 0;
            } else {
            p1++;
            }
            // create a dataype for my pth subrarray in the 3d distribution
            
            
            //d3_array_start holds the starting index of the chunk in the cubes local coordinates(note the cubes local coord system is actually the same as the grids global coord system, by set up)
            
            d3_array_start[x_dim] = d3_coord[x_dim] % cube_sizes[x_dim]; 
            d3_array_start[y_dim] = d3_coord[y_dim] % cube_sizes[y_dim]; 
            d3_array_start[z_dim] = d3_coord[z_dim] % cube_sizes[z_dim]; 
            
            //make starting point so that it coincides with the starting point of the pencil from the pencils coordinate system. (for z_pencils nothing needs to be changed, since it already
            //has the coordinate system of the grid, however, the x and y pencils have different starting points of the subchunk in their coord systems.)
            if(z_dim==0 || z_dim ==1){
            d3_array_start[2]=d3_array_start[2]+subsizes[2]-1;
            }
            if(print_me && (self==me))fprintf(stderr,"D3_array_start is (%d,%d,%d) and subsizes is (%d,%d,%d) \n",d3_array_start[0],d3_array_start[1],d3_array_start[2],subsizes[0],subsizes[1],subsizes[2]);
            
            
            //If sending cube chunks to pencils, need to fill those chunks with data here. The chunks are filled in the order 
            //such that when the pencil recieves the chunk, in its local array indexing, it assumes that the array is already 
            //filled such that it is contiguous. Therefore, complicated for-loops below fill the array in the cubes local indexing to match what the pencil will
            //expect. 
            if(direction == REDISTRIBUTE_3_TO_2){
            int64_t ch_indx=0;
            int dims_size=cube_sizes[0]*cube_sizes[1]*cube_sizes[2];
            if((self == me) && print_me)fprintf(stderr, "%d, %d, MAKE 3D Chunk...\n", self,d3_peer);
            
            #ifdef DIST_OMP
            int last_i;
            #endif
            
            switch(z_dim){
            case 0:
            #ifdef DIST_OMP
            last_i = d3_array_start[y_dim] + 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif
            for(int i2=d3_array_start[y_dim];i2>d3_array_start[y_dim]-subsizes[y_dim];i2--){//perhaps y_dim
                
                #ifdef DIST_OMP
                if(last_i != i2 + 1) {
                ch_indx = 0;
                for(int i=d3_array_start[y_dim]; i>i2; i--)
                for(int i1=d3_array_start[x_dim];i1<d3_array_start[x_dim]+subsizes[x_dim];i1++)
                for(int i0=d3_array_start[z_dim];i0<d3_array_start[z_dim]+subsizes[z_dim];i0++)
                    ch_indx++;
                }
                last_i = i2;
                #endif
                
                for(int i1=d3_array_start[x_dim];i1<d3_array_start[x_dim]+subsizes[x_dim];i1++){//perhaps x_dim
                for(int i0=d3_array_start[z_dim];i0<d3_array_start[z_dim]+subsizes[z_dim];i0++){//perhaps z_dim
                int64_t local_indx=process_topology_3.n[2]*(process_topology_3.n[1]*i0+i1) + i2;
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                d3_chunk[ch_indx]=a[local_indx];
                ch_indx++;
                }
                }
            }
            break;
            case 1:
            #ifdef DIST_OMP
            last_i = d3_array_start[y_dim] - 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif

            for(int i0=d3_array_start[y_dim];i0<d3_array_start[y_dim]+subsizes[y_dim];i0++){
                
                #ifdef DIST_OMP
                if(last_i != i0 - 1) {
                ch_indx = 0;
                for(int i=d3_array_start[y_dim]; i<i0; i++)
                for(int i2=d3_array_start[x_dim];i2>d3_array_start[x_dim]-subsizes[x_dim];i2--)
                for(int i1=d3_array_start[z_dim];i1<d3_array_start[z_dim]+subsizes[z_dim];i1++)
                    ch_indx++;
                }
                last_i = i0;
                #endif

                
                for(int i2=d3_array_start[x_dim];i2>d3_array_start[x_dim]-subsizes[x_dim];i2--){
                for(int i1=d3_array_start[z_dim];i1<d3_array_start[z_dim]+subsizes[z_dim];i1++){
                int64_t local_indx=process_topology_3.n[2]*(process_topology_3.n[1]*i0+i1) + i2;
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                d3_chunk[ch_indx]=a[local_indx];
                ch_indx++;
                }
                }
            }
            
            break;
            case 2:

            #ifdef DIST_OMP
            last_i = d3_array_start[x_dim] - 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif

            for(int i0=d3_array_start[x_dim];i0<d3_array_start[x_dim]+subsizes[x_dim];i0++){
                
                #ifdef DIST_OMP
                if(last_i != i0 - 1) {
                    ch_indx = 0;
                    for(int i=d3_array_start[x_dim]; i<i0; i++)
                for(int i1=d3_array_start[y_dim];i1<d3_array_start[y_dim]+subsizes[y_dim];i1++)
                    for(int i2=d3_array_start[z_dim];i2<d3_array_start[z_dim]+subsizes[z_dim];i2++)
                    ch_indx++;
                }
                last_i = i0;
                #endif

                
                for(int i1=d3_array_start[y_dim];i1<d3_array_start[y_dim]+subsizes[y_dim];i1++){
                for(int i2=d3_array_start[z_dim];i2<d3_array_start[z_dim]+subsizes[z_dim];i2++){
                int64_t local_indx=process_topology_3.n[2]*(process_topology_3.n[1]*i0+i1) + i2;
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                d3_chunk[ch_indx]=a[local_indx];
                ch_indx++;
                }
                }
            }
            
            break;
            }
            }
            
            if (DEBUG_CONDITION || ((self == me) && print_me)) {
            fprintf(stderr,
                "%d: pencil_sizes=(%d,%d,%d), cube_sizes=(%d,%d,%d), subsizes=(%d,%d,%d), d3_coord=(%d,%d,%d), d3_array_start=(%d,%d,%d) \n",
                self,
                pencil_sizes[0], pencil_sizes[1], pencil_sizes[2],
                cube_sizes[0], cube_sizes[1], cube_sizes[2],
                subsizes[0], subsizes[1], subsizes[2],
                d3_coord[0], d3_coord[1], d3_coord[2],
                d3_array_start[0],d3_array_start[1],d3_array_start[2]);
            }
            
            // what peer in the 2d distribution owns this subarray?
            for (int i = 0; i < 3; ++i) {
            switch(z_dim){
            case 0:
            d2_peer_coord[i] = d3_coord[i] / process_topology_2_x.n[i];
            break;
            case 1:
            d2_peer_coord[i] = d3_coord[i] / process_topology_2_y.n[i];
            break;
            case 2:
            d2_peer_coord[i] = d3_coord[i] / process_topology_2_z.n[i];
            break;
            }
            }
            d2_peer_coord[z_dim] = 0;//since these are pencils, there is no two pencils in this direction.
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d PENCIL that hits chunk!...\n",d2_peer_coord[0],d2_peer_coord[1],d2_peer_coord[2]);
            switch(z_dim){
            //find its rank
            case 0:
                rank_x_pencils(&d2_peer,d2_peer_coord);
                break;
            case 1:
                rank_y_pencils(&d2_peer,d2_peer_coord);
                break;
            case 2:
                rank_z_pencils(&d2_peer,d2_peer_coord);
                break;
            }
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d Made it before comm!...\n", self,p, npeers);
            
            // record the communication to be done in a schedule
            if (direction == REDISTRIBUTE_3_TO_2) {
            recv_peer = d3_peer;
            send_peer = d2_peer;
            } else if (direction == REDISTRIBUTE_2_TO_3) {
            recv_peer = d2_peer;
            send_peer = d3_peer;
            } else {
            abort();
            }
            //comunication of the chunks:
            //if print_mess boolean is set to true, then the code runs without sending any messages, and is used to test which messages would be sent in the entire run.
            //(designed to debug comm hangups, if they occur).
            
            if(direction == REDISTRIBUTE_3_TO_2){
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, making request to recieve from %d...\n", self,recv_peer);
            //if(!print_mess)MPI_Irecv((void *) d2_chunk, chunk_size, MPI_DOUBLE_COMPLEX, recv_peer, 0, process_topology_1.cart, &req1);
            if(!print_mess)MPI_Irecv((void *) d2_chunk, chunk_size * sizeof(T), MPI_BYTE, recv_peer, 0, process_topology_1.cart, &req1);
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, making request to send to %d...\n", self,send_peer);
            //if(!print_mess)MPI_Isend((void *) d3_chunk, chunk_size, MPI_DOUBLE_COMPLEX, send_peer, 0, process_topology_1.cart, &req2);
            if(!print_mess)MPI_Isend((void *) d3_chunk, chunk_size * sizeof(T), MPI_BYTE, send_peer, 0, process_topology_1.cart, &req2);
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, waiting to recieve from %d...\n", self,recv_peer);
            //fprintf(stderr, " I am %d, waiting to recieve from %d...\n", self,recv_peer);
            if(!print_mess)MPI_Wait(&req1,MPI_STATUS_IGNORE);
            
            //if((self == me || self == 1 || self == 2 || self == 3) && print_me)fprintf(stderr, " I am %d, waiting to send to %d...\n", self,send_peer);
            //fprintf(stderr, " I am %d, waiting to send to %d...\n", self,send_peer);
            if(self==me && print_mess)fprintf(stderr, " I am %d, waiting to send to %d...\n", self,send_peer);
            if(!print_mess)MPI_Wait(&req2,MPI_STATUS_IGNORE);
            
            //fill the local array with the received chunk.
            int64_t ch_indx=0;
            int dims_size=pencil_dims[0]*pencil_dims[1]*pencil_dims[2];
            if(self==me && print_me)fprintf(stderr,"REAL SUBSIZES (%d,%d,%d)\n",subsizes[x_dim],subsizes[y_dim],subsizes[z_dim]);
            if(self==me && print_me)fprintf(stderr,"PENCIL DIMENSION VS. local sizes (%d,%d,%d) vs (%d,%d,%d)\n",pencil_dims[0],pencil_dims[1],pencil_dims[2],local_sizes[0],local_sizes[1],local_sizes[2]);
            if(self==me && print_me)fprintf(stderr,"DIM_2_ARRAY_START (%d,%d,%d) \n",d2_array_start[0],d2_array_start[1],d2_array_start[2]);
            
            #ifdef DIST_OMP
            int last_i = d2_array_start[0] - 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif

            
            for(int i0=d2_array_start[0];i0<d2_array_start[0]+local_sizes[0];i0++){
            
            #ifdef DIST_OMP
            if(last_i != i0 - 1) {
                ch_indx = 0;
                for(int i=d2_array_start[0]; i<i0; i++)
                for(int i1=d2_array_start[1];i1<d2_array_start[1]+local_sizes[1];i1++)
                    for(int i2=d2_array_start[2];i2<d2_array_start[2]+local_sizes[2];i2++)
                ch_indx++;
            }
            last_i = i0;
            #endif
            
            for(int i1=d2_array_start[1];i1<d2_array_start[1]+local_sizes[1];i1++){
            for(int i2=d2_array_start[2];i2<d2_array_start[2]+local_sizes[2];i2++){
                int64_t local_indx=pencil_dims[2]*(pencil_dims[1]*i0+i1) + i2;
                //if(self==me)fprintf(stderr,"local_indx = %d ",local_indx);
                //if(local_indx >= dims_size)fprintf(stderr,"WOW, in third for, dims is (%d), we are %d and my rank is %d",dims_size,local_indx,self);
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                b[local_indx]=d2_chunk[ch_indx];
                //if((p==0 || p==1 || p==2 || p==3 || p==4 || p==5) && self==me)fprintf(stderr,"(%f,%f) ",real(d2_chunk[ch_indx]),imag(d2_chunk[ch_indx]));
                ch_indx++;
            }
                                    }
            }
            //     if((p==0 ||p==1 || p==2 || p==3 || p==4 || p==5) && self==me)fprintf(stderr,"P is %d \n",p);
            
            } 
            else if (direction == REDISTRIBUTE_2_TO_3) {
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, making request to recieve from %d...\n", self,recv_peer);
            //if(!print_mess)MPI_Irecv((void *) d3_chunk, chunk_size, MPI_DOUBLE_COMPLEX, recv_peer, 0, process_topology_1.cart, &req1);
            if(!print_mess)MPI_Irecv((void *) d3_chunk, chunk_size * sizeof(T), MPI_BYTE, recv_peer, 0, process_topology_1.cart, &req1);
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, making request to send to %d...\n", self,send_peer);
            //if(!print_mess)MPI_Isend((void *) d2_chunk, chunk_size, MPI_DOUBLE_COMPLEX, send_peer, 0, process_topology_1.cart, &req2);
            if(!print_mess)MPI_Isend((void *) d2_chunk, chunk_size * sizeof(T), MPI_BYTE, send_peer, 0, process_topology_1.cart, &req2);
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, waiting to recieve from %d...\n", self,recv_peer);
            if(!print_mess)MPI_Wait(&req1,MPI_STATUS_IGNORE);
            
            if((self == me) && print_mess)fprintf(stderr, " I am %d, waiting to send to %d...\n", self,send_peer);
            if(!print_mess)MPI_Wait(&req2,MPI_STATUS_IGNORE);
            int64_t ch_indx=0;
            int dims_size=(process_topology_3.n[2])*(process_topology_3.n[1])*(process_topology_3.n[0]);
            
            #ifdef DIST_OMP
            int last_i;
            #endif
            
            if(z_dim==0){
            //fill the local array with the received chunk.

            #ifdef DIST_OMP
            last_i = d3_array_start[y_dim] + 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif
            
            for(int i2=d3_array_start[y_dim];i2>d3_array_start[y_dim]-subsizes[y_dim];i2--){
            
            #ifdef DIST_OMP
            if(last_i != i2 + 1) {
            ch_indx = 0;
            for(int i=d3_array_start[y_dim]; i>i2; i--)
                for(int i1=d3_array_start[x_dim];i1<d3_array_start[x_dim]+subsizes[x_dim];i1++)
            for(int i0=d3_array_start[z_dim];i0<d3_array_start[z_dim]+subsizes[z_dim];i0++)
                ch_indx++;
            }
            last_i = i2;
            #endif

            for(int i1=d3_array_start[x_dim];i1<d3_array_start[x_dim]+subsizes[x_dim];i1++){
                for(int i0=d3_array_start[z_dim];i0<d3_array_start[z_dim]+subsizes[z_dim];i0++){
                int64_t local_indx=process_topology_3.n[2]*(process_topology_3.n[1]*i0+i1) + i2;
                //if(local_indx >= dims_size)fprintf(stderr,"WOW, in fourth for, dims is (%d), we are %d and my rank is %d",dims_size,local_indx,self);
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                b[local_indx]=d3_chunk[ch_indx];
                //                         if(p==3 && self==me)fprintf(stderr,"(%f,%f) ",real(d3_chunk[ch_indx]),imag(d3_chunk[ch_indx]));
                ch_indx++;
                }
            }
            }
            }
            else if(z_dim==1){
            
            #ifdef DIST_OMP
            last_i = d3_array_start[y_dim] - 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif

            for(int i0=d3_array_start[y_dim];i0<d3_array_start[y_dim]+subsizes[y_dim];i0++){
            
            #ifdef DIST_OMP
            if(last_i != i0 - 1) {
            ch_indx = 0;
            for(int i=d3_array_start[y_dim]; i<i0; i++)
                for(int i2=d3_array_start[x_dim];i2>d3_array_start[x_dim]-subsizes[x_dim];i2--)
            for(int i1=d3_array_start[z_dim];i1<d3_array_start[z_dim]+subsizes[z_dim];i1++)
                ch_indx++;
            }
            last_i = i0;
            #endif
            
            for(int i2=d3_array_start[x_dim];i2>d3_array_start[x_dim]-subsizes[x_dim];i2--){
                for(int i1=d3_array_start[z_dim];i1<d3_array_start[z_dim]+subsizes[z_dim];i1++){
                int64_t local_indx=process_topology_3.n[2]*(process_topology_3.n[1]*i0+i1) + i2;
                //if(local_indx >= dims_size)fprintf(stderr,"WOW, in fourth for, dims is (%d), we are %d and my rank is %d",dims_size,local_indx,self);
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                b[local_indx]=d3_chunk[ch_indx];
                //                             if(p==0 && self==me)fprintf(stderr,"(%f,%f) ",real(d3_chunk[ch_indx]),imag(d3_chunk[ch_indx]));
                ch_indx++;
                }
            }
            }
            
            }
            else if(z_dim==2){
            
            #ifdef DIST_OMP
            last_i = d3_array_start[x_dim] - 1;
            #pragma omp parallel for schedule(static) firstprivate(ch_indx,last_i)
            #endif

            for(int i0=d3_array_start[x_dim];i0<d3_array_start[x_dim]+subsizes[x_dim];i0++){
            
            #ifdef DIST_OMP
            if(last_i != i0 - 1) {
            ch_indx = 0;
            for(int i=d3_array_start[x_dim]; i<i0; i++)
                for(int i1=d3_array_start[y_dim];i1<d3_array_start[y_dim]+subsizes[y_dim];i1++)
            for(int i2=d3_array_start[z_dim];i2<d3_array_start[z_dim]+subsizes[z_dim];i2++)
                ch_indx++;
            }
            last_i = i0;
            #endif

            for(int i1=d3_array_start[y_dim];i1<d3_array_start[y_dim]+subsizes[y_dim];i1++){
                for(int i2=d3_array_start[z_dim];i2<d3_array_start[z_dim]+subsizes[z_dim];i2++){
                int64_t local_indx=process_topology_3.n[2]*(process_topology_3.n[1]*i0+i1) + i2;
                assert(local_indx < dims_size);
                assert(ch_indx <chunk_size && ch_indx >= 0 && local_indx>=0 && local_indx < dims_size);
                b[local_indx]=d3_chunk[ch_indx];
                //                   if(p==1 && self==me)fprintf(stderr,"(%f,%f) ",real(d3_chunk[ch_indx]),imag(d3_chunk[ch_indx]));
                ch_indx++;
                }
            }
            }
            
            }
            else{
            abort();
            }
            }
            
            if (DEBUG_CONDITION) {
            fprintf(stderr,
                "%d: npeers,p,p0,p1,p1max=(%d,%d,%d,%d,%d), "
                "d3_coord=(%d,%d,%d), d2_peer_coord=(%d,%d,%d), "
                "d2_coord=(%d,%d,%d), d3_peer_coord=(%d,%d,%d), "
                "recv_peer=%d, send_peer=%d\n",
                self,
                npeers, p, p0, p1, p1max,
                d3_coord[0], d3_coord[1], d3_coord[2],
                d2_peer_coord[0], d2_peer_coord[1], d2_peer_coord[2],
                d2_coord[0], d2_coord[1], d2_coord[2],
                d3_peer_coord[0], d3_peer_coord[1], d3_peer_coord[2],
                recv_peer, send_peer);
            }
            
            if((self == me) && print_me)fprintf(stderr, "%d, %d, %d Made it end-for!...\n", self,p, npeers);
        }
        
        //if((self == me) && print_me)fprintf(outfile, "   Made it all the way! for z_dim =(%d) and num_proc = (%d)...\n", z_dim, process_topology_1.nproc[0]);
        if((self == me) && print_result){
            FILE * outfile;
            outfile= fopen("passed.data","a");
            if (outfile) fprintf(outfile, "   Made it all the way! for z_dim =(%d) and num_proc = (%d)...\n", z_dim, process_topology_1.nproc[0]);
            if (outfile) fclose(outfile);
        }
        //    fprintf(stderr, "%d, Made it all the way!...\n", self);

    }


    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::coord_cube(int myrank, int coord[]){
        coord[0] = myrank / (process_topology_3.nproc[1] * process_topology_3.nproc[2]);
        coord[1] = (myrank % (process_topology_3.nproc[1] * process_topology_3.nproc[2]))/(process_topology_3.nproc[2]);
        coord[2] = (myrank % (process_topology_3.nproc[1] * process_topology_3.nproc[2]))%(process_topology_3.nproc[2]);
        return;
    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::rank_cube(int* my_rank, int coord[]){
        *my_rank = coord[2] + (process_topology_3.nproc[2])*(coord[1] + process_topology_3.nproc[1]*coord[0]);
        return;
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::rank_cube(int coord[]){
        int out;
        rank_cube(&out,coord);
        return out;
    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::coord_x_pencils(int myrank, int coord[]){
        // asserts only one processor in x_direction
        assert(process_topology_2_x.nproc[0] == 1);
        //since x_pencils only have one processor in the x_direction.
        coord[0]=0;
        int num_pen_in_cube_col=process_topology_2_x.nproc[1]/process_topology_3.nproc[1];
        int num_pen_in_cube_row=process_topology_2_x.nproc[2]/process_topology_3.nproc[2];
        int num_cubes=(process_topology_3.nproc[2]*process_topology_3.nproc[1]);
        
        /*
        the x_pencil ranks increment in each cube sequencially, after reaching the 
        last cube the second slot in the first cube is the next rank, and then the 
        process repeats. num_repeats, is the number of times this repetition had to 
        have occured to increment to the current rank.
        */
        int num_repeats=myrank/(num_cubes);
        
        //now subtract the difference of how many repetitions, to find the lowest 
        //rank in the cube it resides. 
        int low_rank=myrank-num_repeats*num_cubes;
        
        //find the y and z coords of the low_rank, then adjust coords for ranks 
        //that repeated around the cube.
        coord[1] = (low_rank/process_topology_3.nproc[2])*num_pen_in_cube_col 
            + num_repeats%num_pen_in_cube_col;
        coord[2] = (low_rank%process_topology_3.nproc[2])*num_pen_in_cube_row + num_repeats/num_pen_in_cube_col;
            
        return;

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::rank_x_pencils(int* myrank, int coord[]){
        int num_pen_in_cube_col=process_topology_2_x.nproc[1]/process_topology_3.nproc[1];
        int num_pen_in_cube_row=process_topology_2_x.nproc[2]/process_topology_3.nproc[2];
        if(num_pen_in_cube_col == 0)
            fprintf(stderr,"num_cube_col%d ", 
                process_topology_2_x.nproc[1]/process_topology_3.nproc[1]);
        if(num_pen_in_cube_row ==0)
            fprintf(stderr,"num_cube_row%d ", process_topology_3.nproc[2]);
        assert(num_pen_in_cube_col !=0 && num_pen_in_cube_row !=0);
        int alpha = coord[1]%num_pen_in_cube_col;
        int num_cubes = (process_topology_3.nproc[2]*process_topology_3.nproc[1]);
        int beta = coord[2]%num_pen_in_cube_row;
        *myrank = (alpha*num_cubes) 
            + ((coord[1]/num_pen_in_cube_col)*process_topology_3.nproc[2]) 
            + (beta*(num_cubes)*num_pen_in_cube_col) + coord[2]/num_pen_in_cube_row;
        return;

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::coord_y_pencils(int myrank, int coord[]){
        // asserts only one processor in y_direction
        assert(process_topology_2_y.nproc[1] == 1);
        //since y_pencils only have one processor in the y_direction.
        coord[1] = 0;
        int num_pen_in_cube_row = process_topology_2_y.nproc[2]/process_topology_3.nproc[2];
        int alpha = myrank%(process_topology_2_y.nproc[2]);
        coord[0] = myrank/process_topology_2_y.nproc[2];
        
        coord[2] = (alpha/process_topology_3.nproc[2]) 
            + (alpha%process_topology_3.nproc[2])*num_pen_in_cube_row;
        
        return;

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::rank_y_pencils(int* myrank, int coord[]){
        int num_pen_in_cube_col = process_topology_2_y.nproc[0]/process_topology_3.nproc[0];
        int num_pen_in_cube_row = process_topology_2_y.nproc[2]/process_topology_3.nproc[2];
        //WHY ARE THESE COMMENTED OUT?
        //if(num_pen_in_cube_col ==0)fprintf(stderr,"num_cube_col%d ", process_topology_2_y.nproc[1]/process_topology_3.nproc[1]);
        //if(num_pen_in_cube_row ==0)fprintf(stderr,"num_cube_row%d ", process_topology_3.nproc[2]);
        assert(num_pen_in_cube_col !=0 && num_pen_in_cube_row !=0);
        int beta = coord[2]%num_pen_in_cube_row;
        *myrank = coord[0]*process_topology_2_y.nproc[2] 
            + beta*process_topology_3.nproc[2] 
            + coord[2]/num_pen_in_cube_row;
        return;

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::coord_z_pencils(int myrank, int coord[]){
        // asserts only one processor in z_direction
        assert(process_topology_2_z.nproc[2] == 1);
        //since z_pencils only have one processor in the z_direction.
        coord[2] = 0;
        int num_pen_in_cube_col = process_topology_2_z.nproc[1]/process_topology_3.nproc[1];
        int num_pen_in_cube_row = process_topology_2_z.nproc[0]/process_topology_3.nproc[0];
        int num_pen_in_cube = process_topology_3.nproc[2];
        int alpha = myrank/(process_topology_2_z.nproc[1]*num_pen_in_cube_row);
        coord[0] = alpha*num_pen_in_cube_row + (myrank%num_pen_in_cube)/num_pen_in_cube_col;
        coord[1] = ((myrank%(process_topology_2_z.nproc[1]*num_pen_in_cube_row))/num_pen_in_cube)*num_pen_in_cube_col + myrank%num_pen_in_cube_col;
        
        return;

    }

    template<class T, class MPI_T>
    void distribution_t<T,MPI_T>::rank_z_pencils(int* myrank, int coord[]){
        int num_pen_in_cube_col = process_topology_2_z.nproc[1]/process_topology_3.nproc[1];
        int num_pen_in_cube_row = process_topology_2_z.nproc[0]/process_topology_3.nproc[0];
        int num_pen_in_cube = process_topology_3.nproc[2];
        if(num_pen_in_cube_col == 0)
            fprintf(stderr,"num_cube_col%d ", 
                process_topology_2_z.nproc[1]/process_topology_3.nproc[1]);
        if(num_pen_in_cube_row == 0)
            fprintf(stderr,"num_cube_row%d ", process_topology_3.nproc[2]);
        assert(num_pen_in_cube_col !=0 && num_pen_in_cube_row !=0);
        int alpha = coord[1]%num_pen_in_cube_col;
        int beta = coord[0]%num_pen_in_cube_row;
        *myrank = alpha 
            + ((coord[1]/num_pen_in_cube_col)*num_pen_in_cube) 
            + (beta*num_pen_in_cube_col) 
            + (coord[0]/num_pen_in_cube_row)*process_topology_2_z.nproc[1]*num_pen_in_cube_row;
        return;

    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::rank_x_pencils(int coord[]){
        int out;
        rank_x_pencils(&out,coord);
        return out;
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::rank_y_pencils(int coord[]){
        int out;
        rank_y_pencils(&out,coord);
        return out;
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::rank_z_pencils(int coord[]){
        int out;
        rank_z_pencils(&out,coord);
        return out;
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_nproc_1d(int direction){
        return process_topology_1.nproc[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_nproc_2d_x(int direction){
        return process_topology_2_x.nproc[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_nproc_2d_y(int direction){
        return process_topology_2_y.nproc[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_nproc_2d_z(int direction){
        return process_topology_2_z.nproc[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_nproc_3d(int direction){
        return process_topology_3.nproc[direction];
    }


    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_self_1d(int direction){
        return process_topology_1.self[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_self_2d_x(int direction){
        return process_topology_2_x.self[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_self_2d_y(int direction){
        return process_topology_2_y.self[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_self_2d_z(int direction){
        return process_topology_2_z.self[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::get_self_3d(int direction){
        return process_topology_3.self[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::local_ng_1d(int direction){
        return process_topology_1.n[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::local_ng_2d_x(int direction){
        return process_topology_2_x.n[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::local_ng_2d_y(int direction){
        return process_topology_2_y.n[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::local_ng_2d_z(int direction){
        return process_topology_2_z.n[direction];
    }

    template<class T, class MPI_T>
    int distribution_t<T,MPI_T>::local_ng_3d(int direction){
        return process_topology_3.n[direction];
    }

    template class distribution_t<complexFloatHost,CPUMPI>;
    template class distribution_t<complexDoubleHost,CPUMPI>;

}
}

#endif