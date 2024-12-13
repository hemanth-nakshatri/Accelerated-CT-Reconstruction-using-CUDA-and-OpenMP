#!/usr/bin/env bash
# SBATCH --ntasks=2
# SBATCH --cpus-per-task=2
#SBATCH --partition=instruction 
#SBATCH --time=00:30:00
# SBATCH --nodes=3
#SBATCH --gres=gpu:1
# SBATCH --array=0-3
# SBATCH --gres=gpu:2 -c=3 -N=4
#SBATCH --output=ct_reconstruction.out
#SBATCH --error=ct_reconstruction.err

# Load the CUDA module
module load nvidia/cuda/11.8.0

echo "compiling...";
# Compile the CUDA code
nvcc main.cu ct_processor.cu read_bmp_img.cpp -Xcompiler -O3 -XCompiler -Wall -Xptxas -O3 -fopenmp -std=c++17 -lcufft -lcudart -o ct_reconstruction

echo "Compile completed."

SECONDS = 0
# Default and common parameters
NUM_TRANSDUCERS=180
NUM_ANGLES=180
COMMON_BLOCK_SIZE=256
COMMON_THREADS_PER_DIM=16
INPUT_IMAGE="../images/test1.bmp"
SINOGRAM_IMAGE="../results/test1_sinogram.bmp"
RECONSTRUCTED_IMAGE="../results/test1_reconstructed.bmp"

# Run with common parameters
echo "Running with common parameters..."
./ct_reconstruction $NUM_TRANSDUCERS $NUM_ANGLES $INPUT_IMAGE 4 $COMMON_BLOCK_SIZE $COMMON_THREADS_PER_DIM $SINOGRAM_IMAGE $RECONSTRUCTED_IMAGE

# Run for different block sizes
echo "Running for different block sizes..."
for BLOCK_SIZE in 4 16 32 64 128 256 512 1024; do
    ./ct_reconstruction $NUM_TRANSDUCERS $NUM_ANGLES $INPUT_IMAGE 4 $BLOCK_SIZE $COMMON_THREADS_PER_DIM $SINOGRAM_IMAGE $RECONSTRUCTED_IMAGE
done

# Run for different threads per dimension
echo "Running for different threads per dimension..."
for THREADS_PER_DIM in 1 2 4 8 16 32; do
    ./ct_reconstruction $NUM_TRANSDUCERS $NUM_ANGLES $INPUT_IMAGE 4 $COMMON_BLOCK_SIZE $THREADS_PER_DIM $SINOGRAM_IMAGE $RECONSTRUCTED_IMAGE
done

# Run for varying transducers and angles (optional)
echo "Running for different transducer and angle values..."
for NUM_TRANSDUCERS in 90 180 360; do
    for NUM_ANGLES in 90 180 360; do
        ./ct_reconstruction $NUM_TRANSDUCERS $NUM_ANGLES $INPUT_IMAGE 4 $COMMON_BLOCK_SIZE $COMMON_THREADS_PER_DIM $SINOGRAM_IMAGE $RECONSTRUCTED_IMAGE
    done
done

echo "All executions completed."
echo $SECONDS
