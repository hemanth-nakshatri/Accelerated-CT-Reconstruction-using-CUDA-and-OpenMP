#ifndef CT_PROCESSOR_H
#define CT_PROCESSOR_H

#include <vector>
#include <string>
#include "read_bmp_img.h" // Include the header file containing BMPImage definition

// For Ramp filter
#include <complex>
#include <vector>
#include <cmath>

// Constants
constexpr double PI = 3.14159265358979323846;

struct CTConfig {
    int num_transducers;     // Number of transducers/detectors
    int num_angles;          // Number of projection angles
    double angular_range;    // Total angular range in radians (usually PI or 2*PI)
    
    CTConfig(int transducers = 180, int angles = 180, double range = PI) 
        : num_transducers(transducers), num_angles(angles), angular_range(range) {}
};

class CTProcessor {
private:
    CTConfig config;
    int image_width;
    int image_height;

    // Helper function declarations
    std::vector<unsigned char> normalize_data(const std::vector<double>& data);
    double bilinear_interpolate(const std::vector<unsigned char>& image, 
                              double x, double y, int width, int height);

    // Ramp filter related functions
    // std::vector<double> apply_ramp_filter(const std::vector<double>& sinogram);
    // std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& input);
    // std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& input);
    // void fft_recursive(std::vector<std::complex<double>>& x, bool inverse);

    // CUDA related functions
    

public:
    CTProcessor(const CTConfig& cfg);
    std::vector<double> create_sinogram(const BMPImage& input_image);
    BMPImage reconstruct_image(const std::vector<double>& sinogram);
    void save_sinogram(const std::vector<double>& sinogram, const std::string& filename);

    // Ramp filter related functions
    std::vector<double> apply_ramp_filter(const std::vector<double>& sinogram);
    std::vector<std::complex<double>> fft(const std::vector<std::complex<double>>& input);
    std::vector<std::complex<double>> ifft(const std::vector<std::complex<double>>& input);
    void fft_recursive(std::vector<std::complex<double>>& x, bool inverse);
    
    // CUDA related functions
    std::vector<double> create_sinogram_cuda(const BMPImage& input_image, int block_size);
    BMPImage reconstruct_image_cuda(const std::vector<double>& sinogram, int threads_per_dim);
    std::vector<double> apply_ramp_filter_cuda(const std::vector<double>& sinogram, int block_size);
    };

#endif