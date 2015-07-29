//The MIT License (MIT)
//
//Copyright (c) 2015 Relja Ljubobratovic, ljubobratovic.relja@gmail.com
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.


#include "../include/io.hpp"


extern "C" {
#ifdef WINDOWS
#include "../thirdParty/libpng/include/png.h"
#include "../thirdParty/libpng/include/pngconf.h"
#include "../thirdParty/libpng/include/pnglibconf.h"
}
#else
#include <png.h>
#include <pngconf.h>
#include <pnglibconf.h>
#endif
}


namespace cv {

bool loadPng(cv::image_array &buffer, const std::string &path) {

	int width, height;
	png_byte color_type;
	png_byte bit_depth;

	png_structp png_ptr;
	png_infop info_ptr;
	png_infop end_info;
	png_bytep * row_pointers;

	bool stat = true;

	char header[8];  // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE *fp = fopen(path.c_str(), "rb");
	if (!fp)
		return false;
	fread(header, 1, 8, fp);

	if (png_sig_cmp((png_const_bytep)header, 0, 8))
		return false;

	/* initialize stuff */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr) {
		png_destroy_read_struct(&png_ptr, nullptr, nullptr);
		return false;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
		return false;
	}

	end_info = png_create_info_struct(png_ptr);
	if (!end_info) {
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		return false;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		return false;
	}

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	width = png_get_image_width(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	size_t channels = 0;

	switch (color_type) {
		case PNG_COLOR_TYPE_GRAY:
			channels = 1;
			break;
		case PNG_COLOR_TYPE_GRAY_ALPHA:
			channels = 2;
			break;
		case PNG_COLOR_TYPE_RGB:
			channels = 3;
			break;
		case PNG_COLOR_TYPE_RGBA:
			channels = 4;
			break;
		default:
			png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
			return false;
	}

	png_read_update_info(png_ptr, info_ptr);

	/* read file */
	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		return false;
	}

	if (width && height && bit_depth) {
		data_type dt = NONE;
		switch(bit_depth / 8) {
			case 1:
				dt = UINT8;
				break;
			case 2:
				dt = UINT16;
				break;
			case 4:
				dt = FLOAT32;
				break;
			default:
				std::runtime_error("Invalid bit depth of the image");
		}
		buffer.create(height, width, channels, dt);
	} else {
		std::cerr << "PNG error: failure reading image.\n";
		return false;
	}

	size_t row_ptr_byteCount = png_get_rowbytes(png_ptr, info_ptr);

	row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
	for (int y = 0; y < height; y++)
		row_pointers[y] = (png_byte*)malloc(row_ptr_byteCount);

	png_read_image(png_ptr, row_pointers);

	if (buffer.depth() == 1) {
		byte* data = reinterpret_cast<byte*>(buffer.data());
		for (int y = 0; y < height; y++) {
			byte* row = row_pointers[y];
			memcpy(data, row, buffer.row_stride());
			data += buffer.row_stride();
		}
	} else if (buffer.depth() == 2) {
		unsigned short* data = reinterpret_cast<unsigned short*>(buffer.data());
		for (int y = 0; y < height; y++) {
			unsigned short* row = reinterpret_cast<unsigned short*>(row_pointers[y]);
			for (int x = 0; x < width; x++) {
				unsigned short* ptr = &(row[x * channels]);
				for (unsigned c = 0; c < channels; c++) {
					*(data++) = ptr[c];
				}
			}
		}
	} else {
		std::cerr << "Unsupported png image bit depth.\n";
	}

	/* cleanup heap allocation */
	for (int y = 0; y < height; y++)
		free(row_pointers[y]);

	free(row_pointers);

	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
	fclose(fp);

	return stat;
}

bool writePng(const cv::image_array &buffer, const std::string &path) {

	if (!buffer.is_valid() && buffer.depth() != 1) {
		std::cerr << "Error writing image. Not 8-bit image type.(16-bit writing is not supported)\n";
		return false;
	}

	int bit_depth = buffer.depth() * 8;

	png_structp png_ptr;
	png_infop info_ptr;

	int color_type;

	switch (buffer.channels()) {
		case 1:
			color_type = PNG_COLOR_TYPE_GRAY;
			break;
		case 2:
			color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
			break;
		case 3:
			color_type = PNG_COLOR_TYPE_RGB;
			break;
		case 4:
			color_type = PNG_COLOR_TYPE_RGBA;
	}

	FILE *fp = fopen(path.c_str(), "wb");
	if (!fp) {
		std::cerr << "png write error: cannot open path\n";
		return false;
	}

	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		return false;

	info_ptr = png_create_info_struct(png_ptr);

	if (!info_ptr)
		return false;

	if (setjmp(png_jmpbuf(png_ptr)))
		return false;

	png_init_io(png_ptr, fp);

	/* write header */
	if (setjmp(png_jmpbuf(png_ptr)))
		return false;

	png_set_IHDR(png_ptr, info_ptr, buffer.cols(), buffer.rows(), bit_depth, color_type, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	byte **row_pointers = (byte**)malloc(sizeof(byte*) * buffer.rows());

	byte *data = reinterpret_cast<byte*>(buffer.data());
	size_t rowbufferSize = buffer.cols() * buffer.channels() * buffer.depth();
	for (unsigned y = 0; y < buffer.rows(); y++) {
		row_pointers[y] = (byte*)malloc(rowbufferSize);
	}

	for (unsigned y = 0; y < buffer.rows(); y++) {
		std::copy(data, data + rowbufferSize, row_pointers[y]);
		data += rowbufferSize;
	}

	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr)))
		return false;

	png_write_image(png_ptr, row_pointers);

	/* end write */
	if (setjmp(png_jmpbuf(png_ptr)))
		return false;

	png_write_end(png_ptr, NULL);

	/* cleanup heap allocation */
	for (unsigned y = 0; y < buffer.rows(); y++)
		free(row_pointers[y]);
	free(row_pointers);

	fclose(fp);

	return true;
}


bool imwrite(const image_array &image, const std::string &path) {
	return writePng(image, path);
}

image_array CV_EXPORT imread(const std::string &path, data_type dtype, unsigned channels) {
	image_array im;
	loadPng(im, path);

	if (!im)
		return im;

	if (dtype != NONE)
		im.convert_to(dtype);

	if (channels) {
		switch(channels) {
			case 1:
				im.to_gray();
				break;
			case 3:
				im.to_rgb();
				break;
			case 4:
				im.to_rgba();
				break;
			default:
				throw std::runtime_error("channel count not supported: 1(gray), 3(rgb) and 4(rgba) supported so far.");
		}
	}	

	return im;
}

}
