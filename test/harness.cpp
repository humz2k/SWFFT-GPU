#include "swfft.hpp"
#define CHECK_SILENT
#include "check_kspace.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace SWFFT;

int n_tests = 0;
int n_passed = 0;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define BLOCKSIZE 64

#define avg_of 10

struct benchmark_data_t {

    timing_stats_t initial_f;
    timing_stats_t initial_b;
    timing_stats_t avg_f;
    timing_stats_t avg_b;

};

class output_strings{
    public:
        std::vector<std::string> inis;
        std::vector<std::string> avgs;
    
    output_strings(){};

    ~output_strings(){};

    void print(std::string name){
        int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
        if(world_rank == 0){
            std::cout << std::endl << std::endl << name;
            std::cout << "\n   Initial:\n";
            for (auto & element : inis) std::cout << "      " << element;
            std::cout << "\n   Avg of " << avg_of << ":\n";
            for (auto & element : avgs) std::cout << "      " << element;
            std::cout << std::endl;
        }

    }
};

template<class SWFFT_T, class T>
benchmark_data_t test(bool k_in_blocks, int ngx, int ngy_ = 0, int ngz_ = 0){
    int ngy = ngy_;
    int ngz = ngz_;
    if (ngy == 0){
        ngy = ngx;
    }
    if (ngz == 0){
        ngz = ngx;
    }
    int world_rank; MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
    n_tests++;
    if(world_rank == 0)fprintf(stderr,"Testing %s with T = %s, k_in_blocks = %d and ng = [%d %d %d]\n",typeid(SWFFT_T).name(),typeid(T).name(),k_in_blocks,ngx,ngy,ngz);
    SWFFT_T my_swfft(MPI_COMM_WORLD,ngx,ngy,ngz,BLOCKSIZE,k_in_blocks);
    //my_swfft.query();
    //printf("my_swfft.buff_sz() = %d\n",my_swfft.buff_sz());
    T* data; swfftAlloc(&data,sizeof(T) * my_swfft.buff_sz());
    T* scratch; swfftAlloc(&scratch,sizeof(T) * my_swfft.buff_sz());
    
    bool out = false;

    benchmark_data_t timing_data;
    timing_data.avg_f = {};
    timing_data.avg_b = {};

    //int avg_of = 10;

    for (int i = 0; i < (avg_of + 1); i++){

        assign_delta(data,my_swfft.buff_sz());

        my_swfft.forward_sync(data,scratch);
        //my_swfft.synchronize();

        out = check_kspace(my_swfft,data);

        timing_stats_t forward = my_swfft.getLastTime();

        my_swfft.backward_sync(data,scratch);
        //my_swfft.synchronize();

        out = out && check_rspace(my_swfft,data);

        timing_stats_t backward = my_swfft.getLastTime();

        if (i == 0){
            timing_data.initial_f = forward;
            timing_data.initial_b = backward;
        } else {
            timing_data.avg_f.avg += forward.avg / ((double)avg_of);
            timing_data.avg_f.max += forward.max / ((double)avg_of);
            timing_data.avg_f.min += forward.min / ((double)avg_of);

            timing_data.avg_b.avg += backward.avg / ((double)avg_of);
            timing_data.avg_b.max += backward.max / ((double)avg_of);
            timing_data.avg_b.min += backward.min / ((double)avg_of);
        }

        if(world_rank == 0)fprintf(stderr,".");

    }
    if(world_rank == 0)fprintf(stderr,"\n");
    /*if (world_rank == 0){
        printf("INITIAL TIMES:\n");
        printf("   Forward: avg = %e, max = %e, min = %e\n",timing_data.initial_f.avg,timing_data.initial_f.max,timing_data.initial_f.min);
        printf("   Backward: avg = %e, max = %e, min = %e\n",timing_data.initial_b.avg,timing_data.initial_b.max,timing_data.initial_b.min);
        printf("AVG OF %d FFTS:\n",avg_of);
        printf("   Forward: avg = %e, max = %e, min = %e\n",timing_data.avg_f.avg, timing_data.avg_f.max, timing_data.avg_f.min);
        printf("   Backward: avg = %e, max = %e, min = %e\n",timing_data.avg_b.avg, timing_data.avg_b.max, timing_data.avg_b.min);
    }*/

    if (out){
        n_passed++;
    } else {
        if(world_rank == 0)printf("!!!!!!!!!!!!!Failed!!!!!!!!!!!!\n\n");
    }
    
    swfftFree(data);
    swfftFree(scratch);

    return timing_data;
    //return false;
}

std::string fmt_string(const char* name, const char* type, const char* run_type, timing_stats_t f, timing_stats_t b){
    char fmt[] = "%s[%s]: %s(f: max %6.3e   min %6.3e   mean %6.3e, b: max %6.3e   min %6.3e   mean %6.3e)\n";
    char buffer[500];
    sprintf(buffer,fmt,name,type,run_type,f.max,f.min,f.avg,b.max,b.min,b.avg);
    std::string out = buffer;
    return out;
}

template<class T, class FFTBackend>
output_strings benchmark_cpu(bool k_in_blocks, int ngx, int ngy, int ngz){

    output_strings out;

    #ifdef SWFFT_HQFFT
    benchmark_data_t hqa2a_timings = test<swfft<HQA2ACPU,CPUMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("HQA2ACPU     "," complexDoubleHost ","ini",hqa2a_timings.initial_f,hqa2a_timings.initial_b));
    out.avgs.push_back(fmt_string("HQA2ACPU     "," complexDoubleHost ","avg",hqa2a_timings.avg_f,hqa2a_timings.avg_b));

    benchmark_data_t hqpair_timings = test<swfft<HQPairCPU,CPUMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("HQPairCPU    "," complexDoubleHost ","ini",hqpair_timings.initial_f,hqpair_timings.initial_b));
    out.avgs.push_back(fmt_string("HQPairCPU    "," complexDoubleHost ","avg",hqpair_timings.avg_f,hqpair_timings.avg_b));
    #endif
    #ifdef SWFFT_ALLTOALL
    benchmark_data_t alltoall_timings = test<swfft<AllToAllCPU,CPUMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("AllToAllCPU  "," complexDoubleHost ","ini",alltoall_timings.initial_f,alltoall_timings.initial_b));
    out.avgs.push_back(fmt_string("AllToAllCPU  "," complexDoubleHost ","avg",alltoall_timings.avg_f,alltoall_timings.avg_b));
    #endif
    #ifdef SWFFT_PAIRWISE
    if (!k_in_blocks){
        benchmark_data_t pairwise_timings = test<swfft<Pairwise,CPUMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
        out.inis.push_back(fmt_string("Pairwise     "," complexDoubleHost ","ini",pairwise_timings.initial_f,pairwise_timings.initial_b));
        out.avgs.push_back(fmt_string("Pairwise     "," complexDoubleHost ","avg",pairwise_timings.avg_f,pairwise_timings.avg_b));
    }
    #endif

    return out;

}

template<class T, class FFTBackend>
output_strings benchmark_gpu(bool k_in_blocks, int ngx, int ngy, int ngz){

    output_strings out;

    #ifdef SWFFT_GPU
    #ifdef SWFFT_HQFFT
    benchmark_data_t hqa2a_timings = test<swfft<HQA2AGPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("HQA2AGPU     ","complexDoubleDevice","ini",hqa2a_timings.initial_f,hqa2a_timings.initial_b));
    out.avgs.push_back(fmt_string("HQA2AGPU     ","complexDoubleDevice","avg",hqa2a_timings.avg_f,hqa2a_timings.avg_b));
    benchmark_data_t hqpair_timings = test<swfft<HQPairGPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("HQPairGPU    ","complexDoubleDevice","ini",hqpair_timings.initial_f,hqpair_timings.initial_b));
    out.avgs.push_back(fmt_string("HQPairGPU    ","complexDoubleDevice","avg",hqpair_timings.avg_f,hqpair_timings.avg_b));
    #endif
    #ifdef SWFFT_ALLTOALL
    benchmark_data_t alltoall_timings = test<swfft<AllToAllGPU,OPTMPI,FFTBackend>, T>(k_in_blocks,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("AllToAllGPU  ","complexDoubleDevice","ini",alltoall_timings.initial_f,alltoall_timings.initial_b));
    out.avgs.push_back(fmt_string("AllToAllGPU  ","complexDoubleDevice","avg",alltoall_timings.avg_f,alltoall_timings.avg_b));
    #endif
    #ifdef SWFFT_GPUDELEGATE
    //if(k_in_blocks){
    benchmark_data_t gpudelegate_timings = test<swfft<GPUDelegate,OPTMPI,gpuFFT>,T>(true,ngx,ngy,ngz);
    out.inis.push_back(fmt_string("GPUDelegate  ","complexDoubleDevice","ini",gpudelegate_timings.initial_f,gpudelegate_timings.initial_b));
    out.avgs.push_back(fmt_string("GPUDelegate  ","complexDoubleDevice","avg",gpudelegate_timings.avg_f,gpudelegate_timings.avg_b));
    //}
    #endif    
    #endif

    return out;
}

int main(int argc, char** argv){
    MPI_Init(NULL,NULL);
    #ifdef SWFFT_GPU
    gpuFree(0);
    #endif

    int world_rank;MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);

    if (!((argc == 2) || (argc == 4))){
        if(world_rank == 0)printf("USAGE: %s <ngx> [ngy ngz]\n", argv[0]);
        MPI_Finalize();
        return -1;
    }
    
    int ngx = atoi(argv[1]);
    int ngy = ngx;
    int ngz = ngx;
    if (argc == 4){
        ngy = atoi(argv[2]);
        ngz = atoi(argv[3]);
    }

    swfft_init_threads(2);

    //swfft_init_threads(2);
    #ifdef SWFFT_FFTW
    output_strings cpu_times = benchmark_cpu<complexDoubleHost,fftw>(false,ngx,ngy,ngz);
    #endif

    #if defined(SWFFT_GPU) && defined(SWFFT_CUFFT)
    output_strings gpu_times = benchmark_gpu<complexDoubleDevice,gpuFFT>(false,ngx,ngy,ngz);
    #endif

    cpu_times.print("CPU Times");

    gpu_times.print("GPU Times");
    
    //if(world_rank == 0){
    //    printf("%d/%d tests passed\n",n_passed,n_tests);
    //}
    MPI_Finalize();
    return 0;
}