#ifndef SWFFT_LOGGING_H
#define SWFFT_LOGGING_H

#define SWFFT_LOG

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define CheckCondition(cond) if((!(cond)) && (world_rank == 0)){printf("SWFFT: Failed Check (%s)\n",TOSTRING(cond)); MPI_Abort(MPI_COMM_WORLD,1);}

#ifdef SWFFT_LOG
#define SwfftLog(str,world_rank) {if(world_rank == 0)printf("SWFFT: %s\n",str);}
#else
#define SwfftLog(str,world_rank) {}
#endif

#endif