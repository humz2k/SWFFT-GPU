#!/usr/bin/env bash

total_tests=0
fft="FFTW"
for omp in {TRUE,FALSE}; do
for gpu in {TRUE,FALSE}; do
for dist in {ALLTOALL,PAIRWISE,HQFFT,"ALLTOALL PAIRWISE","ALLTOALL HQFFT","PAIRWISE HQFFT","ALLTOALL PAIRWISE HQFFT"}; do
    total_tests=$((total_tests+1))
done
done
done

gpu="TRUE"
for fft in {FFTW,CUFFT,"FFTW CUFFT"}; do
for omp in {TRUE,FALSE}; do
for dist in {ALLTOALL,PAIRWISE,HQFFT,"ALLTOALL PAIRWISE","ALLTOALL HQFFT","PAIRWISE HQFFT","ALLTOALL PAIRWISE HQFFT"}; do
    total_tests=$((total_tests+1))
done
done
done


RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

counter=0

echo "" > compile.log
echo "" > run.log

test(){
    counter=$((counter+1))
    make clean > /dev/null
    printf "Testing (${counter}/${total_tests}):\n"
    printf "   - Parameters:\n      - USE_OMP=$omp\n      - USE_GPU=$gpu\n      - FFT_BACKEND=\"$fft\"\n      - DIST_BACKEND=\"$dist\"\n"
    echo "" >> compile.log
    echo "" >> compile.log
    echo "############################################" >> compile.log
    echo make -j USE_OMP=$omp USE_GPU=$gpu FFT_BACKEND="$fft" DIST_BACKEND="$dist" >> compile.log
    make -j USE_OMP=$omp USE_GPU=$gpu FFT_BACKEND="$fft" DIST_BACKEND="$dist" &>> compile.log
    if [ $? -eq 0 ]; then
        printf "   - Compile: ${GREEN}PASS${NC}\n"
    else
        printf "   - Compile: ${RED}FAIL${NC}\n"
        exit 1
    fi
    echo "############################################" >> compile.log
    echo "" >> compile.log
    echo "" >> compile.log
    echo "" >> run.log
    echo "" >> run.log
    echo "############################################" >> run.log
    echo make -j USE_OMP=$omp USE_GPU=$gpu FFT_BACKEND="$fft" DIST_BACKEND="$dist" >> run.log
    mpirun -n 8 build/testdfft 8 >> run.log
    if [ $? -eq 0 ]; then
        printf "   - mpirun -n 8 testdfft 8: ${GREEN}PASS${NC}\n"
    else
        printf "   - mpirun -n 8 testdfft 8: ${RED}FAIL${NC}\n"
        exit 1
    fi
    echo "############################################" >> run.log
    echo "" >> run.log
    echo "" >> run.log
}

fft="FFTW"
for omp in {TRUE,FALSE}; do
for gpu in {TRUE,FALSE}; do
for dist in {ALLTOALL,PAIRWISE,HQFFT,"ALLTOALL PAIRWISE","ALLTOALL HQFFT","PAIRWISE HQFFT","ALLTOALL PAIRWISE HQFFT"}; do
    test
done
done
done

gpu="TRUE"
for fft in {FFTW,CUFFT,"FFTW CUFFT"}; do
for omp in {TRUE,FALSE}; do
for dist in {ALLTOALL,PAIRWISE,HQFFT,"ALLTOALL PAIRWISE","ALLTOALL HQFFT","PAIRWISE HQFFT","ALLTOALL PAIRWISE HQFFT"}; do
    test
done
done
done

