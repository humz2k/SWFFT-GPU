#!/bin/bash
#SBATCH -A m4075
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH  --exclusive
#SBATCH  --constraint=gpu
#SBATCH --gpu-bind=none

export SRUN_CPUS_PER_TASK=16
export SLURM_CPU_BIND="cores"
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
srun -n 4 bash -c "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));/global/homes/h/hqureshi/swfft-gpu/build/harness 256"