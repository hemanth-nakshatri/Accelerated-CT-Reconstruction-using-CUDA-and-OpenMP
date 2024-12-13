#include "ct_processor.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <random>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <ctime>

CTProcessor::CTProcessor(const CTConfig& cfg) : config(cfg) {}

// Normalize Data
std::vector<unsigned char> CTProcessor::normalize_data(const std::vector<double>& data) {
    // Find the minimum value in the input data vector
    double min_val = *std::min_element(data.begin(), data.end());
    // Find the maximum value in the input data vector
    double max_val = *std::max_element(data.begin(), data.end());
    
    // Create a vector to hold the normalized values, same size as input data
    std::vector<unsigned char> normalized(data.size());
    
    // Iterate over each element in the input data vector
    for (size_t i = 0; i < data.size(); ++i) {
        // Normalize each element to the range [0, 255] and cast to unsigned char
        normalized[i] = static_cast<unsigned char>(
            255.0 * (data[i] - min_val) / (max_val - min_val)
        );
    }
    
    // Return the vector containing the normalized values
    return normalized;
}

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
template<typename T>
void check_cuda(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\"\n", file, line, 
                static_cast<unsigned int>(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// CUDA kernels
__global__ void convertToGrayscaleKernel(const unsigned char* input, unsigned char* output,
                                        int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        int rgb_idx = idx * 3;
        output[idx] = static_cast<unsigned char>(
            (input[rgb_idx] + input[rgb_idx + 1] + input[rgb_idx + 2]) / 3
        );
    }
}

__global__ void bilinearInterpolateKernel(const unsigned char* image, double* output,
                                         int width, int height, double center_x, double center_y,
                                         double theta, double max_radius, int num_transducers,
                                         int num_samples) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_transducers) return;
    
    double r = (2.0 * t / num_transducers - 1.0) * max_radius;
    double sum = 0.0;
    
    for (int s = 0; s < num_samples; ++s) {
        double position = (2.0 * s / num_samples - 1.0) * max_radius;
        double x = center_x + r * cos(theta) - position * sin(theta);
        double y = center_y + r * sin(theta) + position * cos(theta);
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int x1 = static_cast<int>(floor(x));
            int y1 = static_cast<int>(floor(y));
            int x2 = x1 + 1;
            int y2 = y1 + 1;
            
            if (x2 < width && y2 < height) {
                double fx = x - x1;
                double fy = y - y1;
                
                double c1 = image[y1 * width + x1];
                double c2 = image[y1 * width + x2];
                double c3 = image[y2 * width + x1];
                double c4 = image[y2 * width + x2];
                
                sum += (c1 * (1 - fx) * (1 - fy) +
                       c2 * fx * (1 - fy) +
                       c3 * (1 - fx) * fy +
                       c4 * fx * fy);
            }
        }
    }
    
    output[t] = sum / num_samples;
}

__global__ void applyRampFilterKernel(cufftComplex* data, int padded_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > padded_size/2) return;
    
    float ramp = idx * (2.0f / padded_size);
    data[idx].x *= ramp;
    data[idx].y *= ramp;
    
    if (idx > 0 && idx < padded_size/2) {
        data[padded_size-idx].x *= ramp;
        data[padded_size-idx].y *= ramp;
    }
}

__global__ void backprojectKernel(const float* filtered_sinogram, float* reconstructed,
                                 int width, int height, int num_angles, int num_transducers,
                                 float center_x, float center_y, float max_radius,
                                 float angular_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float px = x - center_x;
    float py = y - center_y;
    float pixel_sum = 0.0f;
    
    for (int angle_idx = 0; angle_idx < num_angles; ++angle_idx) {
        float theta = angle_idx * angular_range / num_angles;
        float r = px * cos(theta) + py * sin(theta);
        float t = ((r / max_radius) + 1.0f) * num_transducers / 2.0f;
        
        int t1 = static_cast<int>(floor(t));
        int t2 = t1 + 1;
        float dt = t - t1;
        
        if (t1 >= 0 && t2 < num_transducers) {
            float val1 = filtered_sinogram[angle_idx * num_transducers + t1];
            float val2 = filtered_sinogram[angle_idx * num_transducers + t2];
            pixel_sum += val1 * (1.0f - dt) + val2 * dt;
        }
    }
    
    reconstructed[y * width + x] = pixel_sum / num_angles;
}

// Updated CTProcessor methods
std::vector<double> CTProcessor::create_sinogram_cuda(const BMPImage& input_image, int block_size) {
    image_width = input_image.width;
    image_height = input_image.height;
    
    // Allocate device memory for input data
    unsigned char *d_input, *d_grayscale;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_image.data.size() * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grayscale, image_width * image_height * sizeof(unsigned char)));
    
    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, input_image.data.data(), 
                               input_image.data.size() * sizeof(unsigned char), 
                               cudaMemcpyHostToDevice));
    
    // Convert to grayscale using CUDA
    // int block_size = 256;
    int num_blocks = (image_width * image_height + block_size - 1) / block_size;
    convertToGrayscaleKernel<<<num_blocks, block_size>>>(d_input, d_grayscale,
                                                        image_width, image_height);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Allocate device memory for sinogram
    double *d_sinogram;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sinogram, 
                               config.num_angles * config.num_transducers * sizeof(double)));
    
    double center_x = image_width / 2.0;
    double center_y = image_height / 2.0;
    double max_radius = std::sqrt(center_x * center_x + center_y * center_y);
    int num_samples = static_cast<int>(2 * max_radius);
    
    // Process each angle
    for (int angle_idx = 0; angle_idx < config.num_angles; ++angle_idx) {
        double theta = angle_idx * config.angular_range / config.num_angles;
        
        num_blocks = (config.num_transducers + block_size - 1) / block_size;
        bilinearInterpolateKernel<<<num_blocks, block_size>>>(
            d_grayscale,
            d_sinogram + angle_idx * config.num_transducers,
            image_width, image_height, center_x, center_y, theta, max_radius,
            config.num_transducers, num_samples
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    
    // Copy result back to host
    std::vector<double> sinogram(config.num_angles * config.num_transducers);
    CHECK_CUDA_ERROR(cudaMemcpy(sinogram.data(), d_sinogram,
                               sinogram.size() * sizeof(double),
                               cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_grayscale);
    cudaFree(d_sinogram);
    
    return sinogram;
}

// Save sinogram as a BMP image
void CTProcessor::save_sinogram(const std::vector<double>& sinogram, const std::string& filename) {
    std::vector<unsigned char> normalized = normalize_data(sinogram);
    
    BMPImage sinogram_image;
    sinogram_image.width = config.num_transducers;
    sinogram_image.height = config.num_angles;
    sinogram_image.data.resize(config.num_transducers * config.num_angles * 3);
    
    for (int i = 0; i < config.num_angles * config.num_transducers; ++i) {
        sinogram_image.data[i * 3] = normalized[i];     // R
        sinogram_image.data[i * 3 + 1] = normalized[i]; // G
        sinogram_image.data[i * 3 + 2] = normalized[i]; // B
    }
    
    // Here you would call your write_bmp function
    write_bmp(filename, sinogram_image);
}

std::vector<double> CTProcessor::apply_ramp_filter_cuda(const std::vector<double>& sinogram, int block_size) {
    int padded_size = 1;
    while (padded_size < config.num_transducers * 2) {
        padded_size *= 2;
    }
    
    try{
    // Allocate memory for CUDA FFT
    cufftHandle plan;
    CHECK_CUDA_ERROR(cufftPlan1d(&plan, padded_size, CUFFT_C2C, 1));
    
    // Allocate device memory
    cufftComplex *d_data;
    float *d_filtered_sinogram;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, padded_size * sizeof(cufftComplex)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filtered_sinogram, 
                               sinogram.size() * sizeof(float)));
    
    // int block_size = 256;
    
    // Process each projection
    for (int angle = 0; angle < config.num_angles; ++angle) {
        // Clear and copy data
        CHECK_CUDA_ERROR(cudaMemset(d_data, 0, padded_size * sizeof(cufftComplex)));
        
        // Convert and copy input data
        std::vector<cufftComplex> h_data(padded_size);
        for (int i = 0; i < config.num_transducers; ++i) {
            h_data[i].x = sinogram[angle * config.num_transducers + i];
            h_data[i].y = 0;
        }
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(),
                                  padded_size * sizeof(cufftComplex),
                                  cudaMemcpyHostToDevice));
        
        // Forward FFT
        CHECK_CUDA_ERROR(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
        
        // Apply ramp filter
        int num_blocks = (padded_size/2 + block_size - 1) / block_size;
        applyRampFilterKernel<<<num_blocks, block_size>>>(d_data, padded_size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Inverse FFT
        CHECK_CUDA_ERROR(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
        
        // Copy back results
        CHECK_CUDA_ERROR(cudaMemcpy(h_data.data(), d_data,
                                  padded_size * sizeof(cufftComplex),
                                  cudaMemcpyDeviceToHost));
        
        // Store the real parts
        for (int i = 0; i < config.num_transducers; ++i) {
            reinterpret_cast<float*>(d_filtered_sinogram)[angle * config.num_transducers + i] = 
                h_data[i].x / padded_size;
        }
    }
    }
    catch (const std::exception& e) {
        std::cerr << "CUDA error: " << e.what() << std::endl;
    }
    
    // Copy final results back to host
    std::vector<double> filtered_sinogram(sinogram.size());
    std::vector<float> temp_result(sinogram.size());
    CHECK_CUDA_ERROR(cudaMemcpy(temp_result.data(), d_filtered_sinogram,
                               sinogram.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    
    // Convert float to double
    for (size_t i = 0; i < sinogram.size(); ++i) {
        filtered_sinogram[i] = temp_result[i];
    }
    
    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    cudaFree(d_filtered_sinogram);
    
    return filtered_sinogram;
}

BMPImage CTProcessor::reconstruct_image_cuda(const std::vector<double>& sinogram, int threads_per_dim) {
    std::vector<double> filtered_sinogram = apply_ramp_filter_cuda(sinogram, threads_per_dim * threads_per_dim);
    
    // Allocate device memory
    float *d_filtered_sinogram, *d_reconstructed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_filtered_sinogram, 
                               filtered_sinogram.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_reconstructed,
                               image_width * image_height * sizeof(float)));
    
    try{
        // Convert and copy filtered sinogram to device
        std::vector<float> float_sinogram(filtered_sinogram.begin(), filtered_sinogram.end());
        CHECK_CUDA_ERROR(cudaMemcpy(d_filtered_sinogram, float_sinogram.data(),
                                filtered_sinogram.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
        
        // Set up grid and block dimensions for 2D kernel
        dim3 block_dim(threads_per_dim, threads_per_dim);
        dim3 grid_dim(
            (image_width + block_dim.x - 1) / block_dim.x,
            (image_height + block_dim.y - 1) / block_dim.y
        );
        
        float center_x = image_width / 2.0f;
        float center_y = image_height / 2.0f;
        float max_radius = std::sqrt(center_x * center_x + center_y * center_y);
        
        // Launch backprojection kernel
        backprojectKernel<<<grid_dim, block_dim>>>(
            d_filtered_sinogram,
            d_reconstructed,
            image_width, image_height, config.num_angles, config.num_transducers,
            center_x, center_y, max_radius, config.angular_range
        );
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Copy result back to host
        std::vector<float> reconstructed(image_width * image_height);
        CHECK_CUDA_ERROR(cudaMemcpy(reconstructed.data(), d_reconstructed,
                                reconstructed.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
        
        // Convert to doubles for normalization
        std::vector<double> double_reconstructed(reconstructed.begin(), reconstructed.end());
        std::vector<unsigned char> normalized = normalize_data(double_reconstructed);
        
        // Create output image
        BMPImage output;
        output.width = image_width;
        output.height = image_height;
        output.data.resize(image_width * image_height * 3);
        
        // Populate BMP image data using OpenMP parallelization
        #pragma omp parallel for
        for (int i = 0; i < image_width * image_height; ++i) {
            output.data[i * 3] = normalized[i];     // R
            output.data[i * 3 + 1] = normalized[i]; // G
            output.data[i * 3 + 2] = normalized[i]; // B
        }

        // Cleanup
        cudaFree(d_filtered_sinogram);
        cudaFree(d_reconstructed);

        // Return or save the output image
        return output;
    }
    catch (const std::exception& e) {
        // Cleanup in case of exception
        cudaFree(d_filtered_sinogram);
        cudaFree(d_reconstructed);
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}