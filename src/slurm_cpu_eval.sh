#!/usr/bin/env zsh
#SBATCH --job-name=CT-Reconstruction
#SBATCH --partition=instruction
#SBATCH --time=00-00:15:00    # Adjusted to allow sufficient time for multiple runs
#SBATCH --ntasks=1            # Single task
#SBATCH --cpus-per-task=2     # CPUs vary dynamically in the script
#SBATCH --nodes=1             # Single node
#SBATCH --output=ct_reconstruction.out
#SBATCH --error=ct_reconstruction.err

echo "Compiling..."
# Compile the CUDA code
g++ -std=c++17 -fopenmp -O3 -o ct_reconstruction main.cpp ct_processor.cpp read_bmp_img.cpp

echo "Compilation completed."

# Execute the program for various configurations
echo "Running scalability analysis..."

# Loop over number of cores
echo "Analyzing performance with varying cores..."
for cores in {1..20}; do
    echo "Cores = $cores"
    ./ct_reconstruction $cores 360 360 ../images/test1.bmp ../results/test1_sinogram_core_${cores}.bmp ../results/test1_reconstructed_core_${cores}.bmp
done

# Loop over number of transducers
echo "Analyzing performance with varying transducers..."
for transducers in {40..360..40}; do
    echo "Transducers = $transducers"
    ./ct_reconstruction 1 $transducers 180 ../images/test1.bmp ../results/test1_sinogram_trans_${transducers}.bmp ../results/test1_reconstructed_trans_${transducers}.bmp
done

# Loop over number of angles
echo "Analyzing performance with varying angles..."
for angles in {40..360..40}; do
    echo "Angles = $angles"
    ./ct_reconstruction 1 180 $angles ../images/test1.bmp ../results/test1_sinogram_angle_${angles}.bmp ../results/test1_reconstructed_angle_${angles}.bmp
done

echo "Scaling analysis completed."
