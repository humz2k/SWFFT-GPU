#ifndef SWFFT_ISENDRECV_SEEN
#define SWFFT_ISENDRECV_SEEN

#include "gpu.hpp"
#include "complex-type.h"
#include "mpiwrangler.hpp"

namespace SWFFT{

    template<class MPI_T, class T>
    class Isend{

    };

    template<class MPI_T, class T>
    class Irecv{

    };

    #ifdef SWFFT_GPU
    template<>
    class Isend<CPUMPI,complexDoubleDevice>{
        private:
            CPUIsend<complexDoubleDevice>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexDoubleDevice>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Isend<CPUMPI,complexFloatDevice>{
        private:
            CPUIsend<complexFloatDevice>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexFloatDevice>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexDoubleDevice>{
        private:
            CPUIrecv<complexDoubleDevice>* raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(CPUIrecv<complexDoubleDevice>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexFloatDevice>{
        private:
            CPUIrecv<complexFloatDevice>* raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(CPUIrecv<complexFloatDevice>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };
    #endif

    template<>
    class Isend<CPUMPI,complexFloatHost>{
        private:
            CPUIsend<complexFloatHost>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexFloatHost>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Isend<CPUMPI,complexDoubleHost>{
        private:
            CPUIsend<complexDoubleHost>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(CPUIsend<complexDoubleHost>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexFloatHost>{
        private:
            CPUIrecv<complexFloatHost>* raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(CPUIrecv<complexFloatHost>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };

    template<>
    class Irecv<CPUMPI,complexDoubleHost>{
        private:
            CPUIrecv<complexDoubleHost>* raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(CPUIrecv<complexDoubleHost>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;}
    };

};

#ifdef SWFFT_GPU
#ifndef SWFFT_NOCUDAMPI

namespace SWFFT{

    //#ifdef SWFFT_GPU
    template<>
    class Isend<GPUMPI,complexDoubleDevice>{
        private:
            GPUIsend<complexDoubleDevice>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(GPUIsend<complexDoubleDevice>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Isend<GPUMPI,complexFloatDevice>{
        private:
            GPUIsend<complexFloatDevice>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(GPUIsend<complexFloatDevice>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Irecv<GPUMPI,complexDoubleDevice>{
        private:
            GPUIrecv<complexDoubleDevice>* raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(GPUIrecv<complexDoubleDevice>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };

    template<>
    class Irecv<GPUMPI,complexFloatDevice>{
        private:
            GPUIrecv<complexFloatDevice>* raw;
        
        public:
            inline Irecv(){};
            
            inline Irecv(GPUIrecv<complexFloatDevice>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };
    //#endif

    template<>
    class Isend<GPUMPI,complexFloatHost>{
        private:
            GPUIsend<complexFloatHost>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(GPUIsend<complexFloatHost>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Isend<GPUMPI,complexDoubleHost>{
        private:
            GPUIsend<complexDoubleHost>* raw;
        
        public:
            inline Isend(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Isend(GPUIsend<complexDoubleHost>* in) : raw(in){};
            inline ~Isend(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();delete raw;};
    };

    template<>
    class Irecv<GPUMPI,complexFloatHost>{
        private:
            GPUIrecv<complexFloatHost>* raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(GPUIrecv<complexFloatHost>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;};
    };

    template<>
    class Irecv<GPUMPI,complexDoubleHost>{
        private:
            GPUIrecv<complexDoubleHost>* raw;
        
        public:
            inline Irecv(){};
            //inline Isend(complexDoubleDevice* in_buff_, int n_, int dest_, int tag_, MPI_Comm comm_) : raw(in_buff_,n_,dest_,tag_,comm_){};
            inline Irecv(GPUIrecv<complexDoubleHost>* in) : raw(in){};
            inline ~Irecv(){};

            inline void execute(){raw->execute();};

            inline void wait(){raw->wait();};

            inline void finalize(){raw->finalize();delete raw;}
    };

};

#endif
#endif

#endif