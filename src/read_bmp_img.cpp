#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include "read_bmp_img.h"

/**
 * Reads a BMP image from the given file path.
 *
 * The function opens the file at the given path in binary mode, reads the
 * headers, and then reads the image data. The image data is stored in a
 * BMPImage object, which is then returned.
 *
 * If the file cannot be opened, a std::runtime_error is thrown.
 *
 * @param filepath the path to the BMP file
 * @return a BMPImage object containing the image data
 */
BMPImage read_bmp(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open BMP file");

    BMPHeader header;
    BMPInfoHeader info_header;

    // Read headers
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.read(reinterpret_cast<char*>(&info_header), sizeof(info_header));

    BMPImage img;
    img.width = info_header.width;
    img.height = info_header.height;
    img.data.resize(img.width * img.height * 3);  // RGB format

    // Move to the pixel data location
    file.seekg(header.offset_data, std::ios::beg);

    // Read pixel data
    file.read(reinterpret_cast<char*>(img.data.data()), img.data.size());

    file.close();
    return img;
}

/**
 * Writes a BMP image to the given file path.
 *
 * The function opens the file at the given path in binary mode, writes the
 * headers, and then writes the image data. The image data is read from the
 * given BMPImage object.
 *
 * If the file cannot be opened, a std::runtime_error is thrown.
 *
 * @param filepath the path to the BMP file
 * @param image the BMPImage object containing the image data
 */
void write_bmp(const std::string& filepath, const BMPImage& image) {
    // Calculate padding for rows (rows must be aligned on 4-byte boundary)
    int padding = (4 - (image.width * 3) % 4) % 4;
    
    // Calculate file and image size
    uint32_t image_size = image.width * image.height * 3 + padding * image.height;
    uint32_t file_size = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + image_size;
    
    // Create and initialize headers
    BMPHeader header;
    header.file_size = file_size;
    header.offset_data = sizeof(BMPHeader) + sizeof(BMPInfoHeader);
    
    BMPInfoHeader info_header;
    info_header.size = sizeof(BMPInfoHeader);
    info_header.width = image.width;
    info_header.height = image.height;
    info_header.bit_count = 24;  // RGB format
    info_header.size_image = image_size;
    
    // Open file for writing
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
    
    // Write headers
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
    
    // Write image data with padding
    std::vector<unsigned char> padding_bytes(padding, 0);
    for (int y = 0; y < image.height; ++y) {
        // Write one row
        file.write(reinterpret_cast<const char*>(&image.data[y * image.width * 3]), 
                  image.width * 3);
        
        // Write padding bytes
        if (padding > 0) {
            file.write(reinterpret_cast<const char*>(padding_bytes.data()), padding);
        }
    }
    
    file.close();
    
    if (!file.good()) {
        throw std::runtime_error("Error occurred while writing file: " + filepath);
    }
}