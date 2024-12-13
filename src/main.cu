#include "ct_processor.h"
#include "read_bmp_img.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <omp.h>
#include <chrono>

int main(int argc, char* argv[]) {
    // Check if the minimum number of arguments is provided
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << "<num_transducers> <num_angles> <input_image> "
                     "[<num_cpu_threads>] [<block_size>] [<threads_per_dim>] [<sinogram_image>] [<reconstructed_image>]"
                  << std::endl;
        return 1;
    }

    // Parse the number of threads, transducers, and angles from command line arguments
    int num_cpu_threads = std::stoi(argv[4]) ? std::stoi(argv[4]) : 1;
    int num_transducers = std::stoi(argv[1]);
    int num_angles = std::stoi(argv[2]);

    try {
        // Configure CT scan parameters with number of transducers, projection angles, and angular range
        CTConfig config(
            num_transducers,  // number of transducers
            num_angles,       // number of projection angles
            PI                // angular range
        );

        // Set the number of threads to be used in parallel sections
        omp_set_num_threads(num_cpu_threads);

        // Initialize the CT processor and measure the initialization time
        auto start = std::chrono::high_resolution_clock::now();
        CTProcessor processor(config);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << "CT Processor time: " << diff.count() << " ms\n";

        // Read the BMP input image and measure the reading time
        start = std::chrono::high_resolution_clock::now();
        BMPImage input_image = read_bmp(argv[3]);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Input Image read time: " << diff.count() << " ms\n";

        // Define block and grid dimensions
        int block_size = std::stoi(argv[5]) ? std::stoi(argv[5]) : 256;
        int threads_per_dim = std::stoi(argv[6]) ? std::stoi(argv[6]) : 16;
        dim3 block_dim(threads_per_dim, threads_per_dim); // For 2D kernels
        dim3 grid_dim(
            (input_image.width + block_dim.x - 1) / block_dim.x,
            (input_image.height + block_dim.y - 1) / block_dim.y
        );

        // Create a sinogram from the input image and measure creation time
        // start = std::chrono::high_resolution_clock::now();
        cudaEvent_t start_sinogram, stop_sinogram;
        cudaEventCreate(&start_sinogram);
        cudaEventCreate(&stop_sinogram);
        cudaEventRecord(start_sinogram);
        std::vector<double> sinogram = processor.create_sinogram_cuda(input_image, block_size);
        // end = std::chrono::high_resolution_clock::now();
        // diff = end - start;
        cudaEventRecord(stop_sinogram);
        cudaEventSynchronize(stop_sinogram);
        float milliseconds_sinogram = 0;
        cudaEventElapsedTime(&milliseconds_sinogram, start_sinogram, stop_sinogram);
        cudaDeviceSynchronize();
        std::cout << "Sinogram creation time: " << milliseconds_sinogram << " ms\n";

        // Save the sinogram to a file and measure the saving time
        start = std::chrono::high_resolution_clock::now();
        std::string sinogram_image = (argc > 7) ? argv[7] : "sinogram.bmp"; // Default to "sinogram.bmp" if not provided
        processor.save_sinogram(sinogram, sinogram_image);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        cudaDeviceSynchronize();
        std::cout << "Save Sinogram time: " << diff.count() << " ms\n";

        // Apply the ramp filter
        // start = std::chrono::high_resolution_clock::now();
        cudaEvent_t start_ramp, stop_ramp;
        cudaEventCreate(&start_ramp);
        cudaEventCreate(&stop_ramp);
        cudaEventRecord(start_ramp);
        std::vector<double> filtered_sinogram = processor.apply_ramp_filter_cuda(sinogram, block_size);
        processor.save_sinogram(filtered_sinogram, sinogram_image);
        cudaEventRecord(stop_ramp);
        cudaEventSynchronize(stop_ramp);
        float milliseconds_ramp = 0;
        cudaEventElapsedTime(&milliseconds_sinogram, start_ramp, stop_ramp);
        cudaDeviceSynchronize();
        // end = std::chrono::high_resolution_clock::now();
        // diff = end - start;
        std::cout << "Ramp Filter time: " << milliseconds_ramp << " ms\n";

        // Reconstruct the image from the filtered sinogram and measure reconstruction time
        start = std::chrono::high_resolution_clock::now();
        BMPImage reconstructed = processor.reconstruct_image_cuda(filtered_sinogram, threads_per_dim);
        // BMPImage reconstructed = processor.reconstruct_image_cuda(sinogram, threads_per_dim);
	    end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        cudaDeviceSynchronize();
        std::cout << "CT Reconstruction time: " << diff.count() << " ms\n";

        // Save the reconstructed image to a file and measure the saving time
        start = std::chrono::high_resolution_clock::now();
        std::string reconstructed_image = (argc > 8) ? argv[8] : "reconstructed.bmp"; // Default to "reconstructed.bmp" if not provided
        write_bmp(reconstructed_image, reconstructed);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        cudaDeviceSynchronize();
        std::cout << "Save reconstruction time: " << diff.count() << " ms\n";

    } catch (const std::exception& e) {
        // Catch and print any exceptions that occur during processing
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Return success
    return 0;
}
