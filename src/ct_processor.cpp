#include "ct_processor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

CTProcessor::CTProcessor(const CTConfig& cfg) : config(cfg) {}

/**
 * Normalize a vector of double data to a range of 0 to 255.
 * This is usually to normalize the image data.
 *
 * This function takes a vector of double-precision values and normalizes each value
 * such that the minimum value in the input vector maps to 0 and the maximum value maps
 * to 255. The result is a vector of unsigned char values representing the normalized data.
 *
 * @param data A vector of double values to be normalized.
 * @return A vector of unsigned char values representing the normalized data.
 */
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

/**
 * Bilinearly interpolate a pixel value from an image at a given point.
 *
 * The given point (x, y) is not necessarily a pixel location in the image.
 * Instead, this function will interpolate the value between the four
 * surrounding pixels to produce a value at the given point.
 *
 * The coordinates (x, y) are floating-point values, and can be any valid
 * double-precision number. The function will return 0.0 if the point is
 * outside of the image.
 *
 * @param image A vector of unsigned char values representing the image data.
 * @param x The x-coordinate of the point to be interpolated.
 * @param y The y-coordinate of the point to be interpolated.
 * @param width The width of the image.
 * @param height The height of the image.
 * @return The interpolated value at the point (x, y), or 0.0 if the point is
 * outside of the image.
 */
double CTProcessor::bilinear_interpolate(const std::vector<unsigned char>& image, 
                                       double x, double y, int width, int height) {

    // First, calculate the coordinates of the four surrounding pixels.
    // These are the pixels that we will use to interpolate the value at (x, y).

    int x1 = static_cast<int>(std::floor(x));
    int y1 = static_cast<int>(std::floor(y));
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // If any of the surrounding pixels are outside of the image, return 0.0.
    // This is because we can't interpolate values if we don't have any valid
    // values to interpolate from.
    if (x1 < 0 || x2 >= width || y1 < 0 || y2 >= height) {
        return 0.0;
    }

    // Calculate the fractional parts of x and y. These are the values that we
    // will use to determine the weights of each pixel in the interpolation
    // process.
    double fx = x - x1;
    double fy = y - y1;

    // Calculate the values of the four surrounding pixels. These are the values
    // that we will use to interpolate the value at (x, y).
    double c1 = image[y1 * width + x1];
    double c2 = image[y1 * width + x2];
    double c3 = image[y2 * width + x1];
    double c4 = image[y2 * width + x2];

    // Calculate the interpolated value at (x, y). This is done by
    // multiplying each of the surrounding pixels by the weight of that pixel
    // and then summing the results. The weights are calculated as follows:
    // - The weight of the top-left pixel is (1 - fx) * (1 - fy)
    // - The weight of the top-right pixel is fx * (1 - fy)
    // - The weight of the bottom-left pixel is (1 - fx) * fy
    // - The weight of the bottom-right pixel is fx * fy
    //
    // These weights are then used to calculate the final interpolated value.
    return (c1 * (1 - fx) * (1 - fy) +
            c2 * fx * (1 - fy) +
            c3 * (1 - fx) * fy +
            c4 * fx * fy);
}

/**
 * Create a sinogram from a given BMP image.
 * 
 * The function converts the input RGB image to a grayscale image and then
 * performs the Radon transform to create a sinogram. The sinogram is a 
 * 2D representation of the 1D projections of the image at various angles.
 * 
 * The process involves bilinear interpolation for computing the Radon 
 * transform at non-integer coordinates within the image.
 * 
 * @param input_image The BMP image to be processed, containing RGB data.
 * @return A vector of double values representing the sinogram, with
 *         dimensions [num_angles x num_transducers].
 */
std::vector<double> CTProcessor::create_sinogram(const BMPImage& input_image) {
    // Store image dimensions
    image_width = input_image.width;
    image_height = input_image.height;
    
    // Initialize a vector to store the grayscale values of the image
    std::vector<unsigned char> grayscale(image_width * image_height);
    
    // Loop over each pixel in the input image
    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < image_width; ++j) {
            // Calculate the index for the RGB components of the pixel
            int idx = (i * image_width + j) * 3;
            
            // Convert the RGB pixel to grayscale by averaging the R, G, and B components
            grayscale[i * image_width + j] = static_cast<unsigned char>(
                (input_image.data[idx] + input_image.data[idx + 1] + input_image.data[idx + 2]) / 3
            );
        }
    }

    // Initialize the sinogram, a 2D array flattened into a 1D vector
    std::vector<double> sinogram(config.num_angles * config.num_transducers, 0.0);
    
    // Determine the center of the image
    double center_x = image_width / 2.0;
    double center_y = image_height / 2.0;
    
    // Calculate the maximum radius from the center to the corners of the image
    double max_radius = std::sqrt(center_x * center_x + center_y * center_y);
    
    // Loop over each angle and transducer position to calculate the Radon transform
    #pragma omp parallel for collapse(2)
    for (int angle_idx = 0; angle_idx < config.num_angles; ++angle_idx) {
        for (int t = 0; t < config.num_transducers; ++t) {
            // Calculate the angle and radial distance for the current detector position
            double theta = angle_idx * config.angular_range / config.num_angles;
            double r = (2.0 * t / config.num_transducers - 1.0) * max_radius;
            
            // Initialize the sum for the Radon transform at this angle and position
            double sum = 0.0;
            
            // Number of samples taken along the radial line
            int num_samples = static_cast<int>(2 * max_radius);
            
            // Loop over each sample position along the radial line
            for (int s = 0; s < num_samples; ++s) {
                // Calculate the position along the radial line
                double position = (2.0 * s / num_samples - 1.0) * max_radius;
                
                // Calculate the x and y coordinates in the image for the current position
                double x = center_x + r * std::cos(theta) - position * std::sin(theta);
                double y = center_y + r * std::sin(theta) + position * std::cos(theta);
                
                // If the coordinates are within the image boundaries, perform interpolation
                if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
                    // Add the interpolated grayscale value to the sum
                    sum += bilinear_interpolate(grayscale, x, y, image_width, image_height);
                }
            }
            
            // Store the normalized sum in the sinogram
            sinogram[angle_idx * config.num_transducers + t] = sum / num_samples;
        }
    }
    
    // Return the computed sinogram
    return sinogram;
}

/**
 * Saves a sinogram to a BMP file.
 *
 * This function normalizes the input sinogram data and constructs a BMP 
 * image where each pixel's RGB channels are set to the normalized sinogram 
 * value. The resulting image is stored in a file specified by the given 
 * filename.
 *
 * @param sinogram A vector of double values representing the sinogram 
 *                 data to be saved, with dimensions [num_angles x num_transducers].
 * @param filename A string specifying the name of the file where the 
 *                 sinogram image will be saved.
 */
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

// Perform a recursive Fast Fourier Transform (FFT) or Inverse FFT on a vector of complex numbers
void CTProcessor::fft_recursive(std::vector<std::complex<double>>& x, bool inverse) {
    const size_t N = x.size(); // Get the size of the input vector
    if (N <= 1) return; // Base case: If the vector size is 1 or less, no further processing is needed

    // Split the input vector into two halves: even-indexed and odd-indexed elements
    std::vector<std::complex<double>> even(N/2), odd(N/2);
    for (size_t i = 0; i < N/2; i++) {
        even[i] = x[2*i]; // Collect even-indexed elements
        odd[i] = x[2*i+1]; // Collect odd-indexed elements
    }

    // Recursively apply FFT on both halves
    fft_recursive(even, inverse);
    fft_recursive(odd, inverse);

    // Combine the results of the recursive FFT
    double angle_multiplier = inverse ? 2*PI/N : -2*PI/N; // Determine angle based on whether it's FFT or inverse FFT
    std::complex<double> w(1), wn(std::cos(angle_multiplier), std::sin(angle_multiplier)); // Initialize twiddle factors
    
    for (size_t i = 0; i < N/2; i++) {
        x[i] = even[i] + w * odd[i]; // Combine even and odd parts
        x[i + N/2] = even[i] - w * odd[i]; // Combine even and odd parts with a negative sign

        if (inverse) { // If it's an inverse FFT, scale the results
            x[i] /= 2;
            x[i + N/2] /= 2;
        }
        
        w *= wn; // Update the twiddle factor
    }
}

std::vector<std::complex<double>> CTProcessor::fft(const std::vector<std::complex<double>>& input) {
    std::vector<std::complex<double>> result = input;
    fft_recursive(result, false);
    return result;
}

std::vector<std::complex<double>> CTProcessor::ifft(const std::vector<std::complex<double>>& input) {
    std::vector<std::complex<double>> result = input;
    fft_recursive(result, true);
    return result;
}

/**
 * Applies a ramp filter to the given sinogram.
 *
 * The ramp filter is a filter in the frequency domain that multiplies
 * the magnitude of the frequency components by their frequency. The
 * filter is applied to each projection angle separately, and the
 * filtered sinogram is returned as a vector of double values.
 *
 * @param sinogram The input sinogram, a vector of double values with
 *                 dimensions [num_angles x num_transducers].
 * @return The filtered sinogram, a vector of double values with the
 *         same dimensions as the input.
 */
std::vector<double> CTProcessor::apply_ramp_filter(const std::vector<double>& sinogram) {
    const int num_detectors = config.num_transducers;
    
    // Find next power of 2 for FFT
    int padded_size = 1;
    while (padded_size < num_detectors * 2) {
        padded_size *= 2;
    }
    
    // Create ramp filter in frequency domain
    std::vector<double> ramp(padded_size/2 + 1);
    for (int i = 0; i < ramp.size(); ++i) {
        // The ramp filter is a simple filter that multiplies the
        // magnitude of the frequency components by their frequency.
        // The factor of 2.0 is because the filter is applied to both
        // positive and negative frequencies.
        ramp[i] = i * (2.0 / padded_size);
    }
    
    // Process each projection angle
    std::vector<double> filtered_sinogram(sinogram.size());
    
    // Use OpenMP to parallelize the loop over the projection angles
    #pragma omp parallel for
    for (int angle = 0; angle < config.num_angles; ++angle) {
        // Prepare input for FFT
        std::vector<std::complex<double>> padded_projection(padded_size);
        for (int i = 0; i < num_detectors; ++i) {
            // Copy the input sinogram to a padded vector, which is
            // necessary for the FFT algorithm.
            padded_projection[i] = sinogram[angle * num_detectors + i];
        }
        
        // Apply FFT to the padded vector
        auto freq_domain = fft(padded_projection);
        
        // Apply ramp filter in frequency domain
        for (int i = 0; i < padded_size/2 + 1; ++i) {
            // Multiply the frequency components by the ramp filter
            freq_domain[i] *= ramp[i];
            
            // The ramp filter is symmetric, so we can apply it to both
            // positive and negative frequencies by multiplying the
            // conjugate of the frequency components by the ramp filter.
            if (i > 0 && i < padded_size/2) {
                freq_domain[padded_size-i] *= ramp[i];
            }
        }
        
        // Apply inverse FFT to get the filtered projection
        auto filtered = ifft(freq_domain);
        
        // Copy back the filtered projection to the output vector
        for (int i = 0; i < num_detectors; ++i) {
            filtered_sinogram[angle * num_detectors + i] = filtered[i].real();
        }
    }
    
    return filtered_sinogram;
}

/**
 * Reconstruct an image from a sinogram.
 * 
 * The function applies a ramp filter to the input sinogram and then
 * performs a backprojection to reconstruct the image. The image is
 * represented as a 2D array of double values, with each pixel value
 * representing the intensity of the pixel in the image.
 * 
 * The function uses OpenMP to parallelize the loop over the projection
 * angles and the loop over the image pixels. This makes the function
 * much faster than a naive implementation.
 * 
 * @param sinogram The input sinogram, which is a 2D vector of double
 *                 values representing the line integrals of the image
 *                 at different angles and transducer positions.
 * @return A 2D vector of double values representing the reconstructed
 *         image.
 */
BMPImage CTProcessor::reconstruct_image(const std::vector<double>& filtered_sinogram) {

    // Apply ramp filter to sinogram
    // std::vector<double> filtered_sinogram = apply_ramp_filter(sinogram);

    std::vector<double> reconstructed(image_width * image_height, 0.0);
    double center_x = image_width / 2.0;
    double center_y = image_height / 2.0;
    double max_radius = std::sqrt(center_x * center_x + center_y * center_y);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
            double px = x - center_x;
            double py = y - center_y;
            double pixel_sum = 0.0;
            
            for (int angle_idx = 0; angle_idx < config.num_angles; ++angle_idx) {
                double theta = angle_idx * config.angular_range / config.num_angles;
                double r = px * std::cos(theta) + py * std::sin(theta);
                double t = ((r / max_radius) + 1.0) * config.num_transducers / 2.0;
                
                int t1 = static_cast<int>(std::floor(t));
                int t2 = t1 + 1;
                double dt = t - t1;
                
                if (t1 >= 0 && t2 < config.num_transducers) {
                    double val1 = filtered_sinogram[angle_idx * config.num_transducers + t1];
                    double val2 = filtered_sinogram[angle_idx * config.num_transducers + t2];
                    pixel_sum += val1 * (1 - dt) + val2 * dt;
                }
            }
            
            reconstructed[y * image_width + x] = pixel_sum / config.num_angles;
        }
    }
    
    std::vector<unsigned char> normalized = normalize_data(reconstructed);
    
    BMPImage output;
    output.width = image_width;
    output.height = image_height;
    output.data.resize(image_width * image_height * 3);
    
    #pragma omp parallel for
    for (int i = 0; i < image_width * image_height; ++i) {
        output.data[i * 3] = normalized[i];     // R
        output.data[i * 3 + 1] = normalized[i]; // G
        output.data[i * 3 + 2] = normalized[i]; // B
    }
    
    return output;
}
