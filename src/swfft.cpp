#include "swfft.hpp"

template<class Backend>
swfft<Backend>::swfft(int ngx, int ngy, int ngz, int blockSize, MPI_Comm comm);