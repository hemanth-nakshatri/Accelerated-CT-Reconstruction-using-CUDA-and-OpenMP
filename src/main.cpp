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
        std::cerr << "Usage: " << argv[0] << " <num_threads> <num_transducers> <num_angles> <input_image> [<sinogram_image> [<reconstructed_image>] ]" << std::endl;
        return 1;
    }

    // Parse the number of threads, transducers, and angles from command line arguments
    int num_threads = std::stoi(argv[1]);
    int num_transducers = std::stoi(argv[2]);
    int num_angles = std::stoi(argv[3]);

    try {
        // Configure CT scan parameters with number of transducers, projection angles, and angular range
        CTConfig config(
            num_transducers,  // number of transducers
            num_angles,       // number of projection angles
            PI                // angular range
        );

        // Set the number of threads to be used in parallel sections
        omp_set_num_threads(num_threads);
        
        // Initialize the CT processor and measure the initialization time
        auto start = std::chrono::high_resolution_clock::now();
        CTProcessor processor(config);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << "CT Processor time: " << diff.count() << " ms\n";

        // Read the BMP input image and measure the reading time
        start = std::chrono::high_resolution_clock::now();
        BMPImage input_image = read_bmp(argv[4]);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Input Image read time: " << diff.count() << " ms\n";
        
        // Create a sinogram from the input image and measure creation time
        start = std::chrono::high_resolution_clock::now();
        std::vector<double> sinogram = processor.create_sinogram(input_image);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Sinogram creation time: " << diff.count() << " ms\n";
        
        // Save the sinogram to a file and measure the saving time
        start = std::chrono::high_resolution_clock::now();
        std::string sinogram_image = (argc > 5) ? argv[5] : "sinogram.bmp"; // Default to "sinogram.bmp" if not provided
        processor.save_sinogram(sinogram, sinogram_image);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Save Sinogram time: " << diff.count() << " ms\n";

        // Apply the ramp filter
        start = std::chrono::high_resolution_clock::now();
        std::vector<double> filtered_sinogram = processor.apply_ramp_filter(sinogram);
        processor.save_sinogram(filtered_sinogram, sinogram_image);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Ramp Filter time: " << diff.count() << " ms\n";
        
        // Reconstruct the image from the sinogram and measure reconstruction time
        start = std::chrono::high_resolution_clock::now();
        BMPImage reconstructed = processor.reconstruct_image(filtered_sinogram);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "CT Reconstruction time: " << diff.count() << " ms\n";
        
        // Save the reconstructed image to a file and measure the saving time
        start = std::chrono::high_resolution_clock::now();
        std::string reconstructed_image = (argc > 6) ? argv[6] : "reconstructed.bmp"; // Default to "reconstructed.bmp" if not provided
        write_bmp(reconstructed_image, reconstructed);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        std::cout << "Save reconstruction time: " << diff.count() << " ms\n";
        
    } catch (const std::exception& e) {
        // Catch and print any exceptions that occur during processing
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Return success
    return 0;
}
