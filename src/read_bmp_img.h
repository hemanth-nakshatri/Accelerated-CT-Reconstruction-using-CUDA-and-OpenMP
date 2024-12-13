#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

#ifndef READ_BMP_IMG_H
#define READ_BMP_IMG_H

#pragma pack(push, 1)  // Ensure no padding in structs
struct BMPHeader {
    uint16_t file_type{0x4D42};   // File type always BM which is 0x4D42
    uint32_t file_size{0};         // Size of the file (in bytes)
    uint16_t reserved1{0};         // Reserved
    uint16_t reserved2{0};         // Reserved
    uint32_t offset_data{0};       // Start position of pixel data (bytes from the beginning of the file)
};

struct BMPInfoHeader {
    uint32_t size{0};              // Size of this header (in bytes)
    int32_t width{0};              // Width of bitmap in pixels
    int32_t height{0};             // Height of bitmap in pixels
    uint16_t planes{1};            // No. of planes for the target device, this is always 1
    uint16_t bit_count{0};         // No. of bits per pixel
    uint32_t compression{0};       // 0 or 3 - uncompressed
    uint32_t size_image{0};        // 0 - for uncompressed images
    int32_t x_pixels_per_meter{0};
    int32_t y_pixels_per_meter{0};
    uint32_t colors_used{0};       // No. of colors used for the bitmap
    uint32_t colors_important{0};  // No. of important colors
};
#pragma pack(pop)

struct BMPImage {
    int width;
    int height;
    std::vector<unsigned char> data;
};

// Function declaration
BMPImage read_bmp(const std::string& filepath);
void write_bmp(const std::string& filepath, const BMPImage& image);

#endif