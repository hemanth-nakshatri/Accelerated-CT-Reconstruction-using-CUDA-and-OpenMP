#!/usr/bin/env zsh
#SBATCH --job-name=CT-Reconstruction
#SBATCH --partition=instruction
#SBATCH --time=00-00:01:00
# SBATCH --ntasks=2
# SBATCH --cpus-per-task=2
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
nvcc main.cu ct_processor.cu read_bmp_img.cpp -Xcompiler -fopenmp -std=c++17 -lcufft -lcudart -o ct_reconstruction

echo "compile completed?";
# Run the program with sample arguments
./ct_reconstruction 360 360  ../images/test1.bmp 4 512 32 ../results/test1_sinogram.bmp ../results/test1_reconstructed.bmp

# Scaling analysis
# Analyze performance with varying input sizes and configurations
# Uncomment and modify as needed

# for i in {5..14}; do 
#     echo "Run $i: Size = $((2**i)), Num Angles = 180";
#     ./ct_reconstruction 4 256 180 input_$((2**i)).bmp sinogram_$((2**i)).bmp reconstructed_$((2**i)).bmp;
# done;
