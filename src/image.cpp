// The MIT License (MIT)
// 
// Copyright (c) 2015 Relja Ljubobratovic, ljubobratovic.relja@gmail.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "../include/image.hpp"


namespace cv {

namespace internal {
template<typename _Tp>
void cv_to_gray(const image_array &source, image_array &target) {
    for (unsigned i = 0; i < source.rows(); ++i) {
        for (unsigned j = 0; j < source.cols(); ++j) {
            target.at<_Tp>(i, j) = static_cast<_Tp>((source.at<_Tp>(i, j, 0) +
                                                    source.at<_Tp>(i, j, 1) +
                                                    source.at<_Tp>(i, j, 2)) / 3.);
        }
    }
}


template<typename _Tp>
void cv_to_rgb(const image_array &source, image_array &target) {
    if (source.channels() == 1) {
        for (unsigned i = 0; i < source.rows(); ++i) {
            for (unsigned j = 0; j < source.cols(); ++j) {
                target.at<_Tp>(i, j, 0) = source.at<_Tp>(i, j);
                target.at<_Tp>(i, j, 1) = source.at<_Tp>(i, j);
                target.at<_Tp>(i, j, 2) = source.at<_Tp>(i, j);
            }
        }
    } else if (source.channels() == 4) {
        for (unsigned i = 0; i < source.rows(); ++i) {
            for (unsigned j = 0; j < source.cols(); ++j) {
                target.at<_Tp>(i, j, 0) = source.at<_Tp>(i, j, 0);
                target.at<_Tp>(i, j, 1) = source.at<_Tp>(i, j, 1);
                target.at<_Tp>(i, j, 2) = source.at<_Tp>(i, j, 2);
            }
        }
    } else {
        std::cerr << "Invalid source image!~ image should be a RGB, or RGBA to be converted to gray.\n";
    }
}

template<typename _Tp>
void cv_to_rgba(const image_array &source, image_array &target) {
    if (source.channels() == 1) {
        for (unsigned i = 0; i < source.rows(); ++i) {
            for (unsigned j = 0; j < source.cols(); ++j) {
                target.at<_Tp>(i, j, 0) = source.at<_Tp>(i, j);
                target.at<_Tp>(i, j, 1) = source.at<_Tp>(i, j);
                target.at<_Tp>(i, j, 2) = source.at<_Tp>(i, j);
                target.at<_Tp>(i, j, 3) = 255;
            }
        }
    } else if (source.channels() == 3) {
        for (unsigned i = 0; i < source.rows(); ++i) {
            for (unsigned j = 0; j < source.cols(); ++j) {
                target.at<_Tp>(i, j, 0) = source.at<_Tp>(i, j, 0);
                target.at<_Tp>(i, j, 1) = source.at<_Tp>(i, j, 1);
                target.at<_Tp>(i, j, 2) = source.at<_Tp>(i, j, 2);
                target.at<_Tp>(i, j, 3) = 255;
            }
        }
    }
}
}

///////////////////////////////////////////////////////////////////////////////
/// class: image_array

image_array::image_array(): super_type(), _dtype(NONE) {
}

image_array::image_array(unsigned rows, unsigned cols, unsigned channels, data_type dtype) : super_type(), _dtype(dtype) {
    ASSERT(dtype != NONE);
    this->_refcount = REF_NEW;
    this->allocate({rows, cols, channels}, internal::data_size[dtype]);
}

image_array::image_array(const vector<image_array> &channels) : super_type() {
	this->merge(channels);
}

image_array::image_array(const image_array &cpy, bool deep_copy) : super_type(cpy, deep_copy), _dtype(cpy._dtype) {
}

image_array::image_array(image_array &&move) {
    this->move(std::move(move));
    this->_dtype = move._dtype;
    move._dtype = NONE;
}

image_array::~image_array() {
}

image_array &image_array::operator=(image_array &&rhs) {
    if (this != &rhs) {
        super_type::operator=(std::move(rhs));
        this->_dtype = rhs._dtype;
        rhs._dtype = NONE;
    }
    return *this;
}

void image_array::create(unsigned rows, unsigned cols, unsigned channels, data_type dtype) {
    this->release();
    this->_dtype = dtype;
    ASSERT(dtype != NONE);
    this->allocate({rows, cols, channels}, internal::data_size[dtype]);
}

image_array image_array::clone() const {
    return image_array(*this, true);
}


unsigned image_array::rows() const {
    return this->_shape[0];
}

unsigned image_array::cols() const {
    return this->_shape[1];
}

unsigned image_array::channels() const {
    return this->_shape[2];
}

void image_array::release() {
    super_type::release();
    this->_dtype = NONE;
}

vec3i image_array::size() const {
    return vec3i(this->rows(), this->cols(), this->channels());
}

data_type image_array::dtype() const {
    return this->_dtype;
}

unsigned image_array::row_stride() const {
    return this->_strides[0];
}

unsigned image_array::depth() const {
    return internal::data_size[this->_dtype];
}

vector<image_array> image_array::split() {
	ASSERT(*this);

	if (this->channels() == 1)
		return {*this};
	
	vector<image_array> channels(this->channels());

	for (unsigned i = 0; i < this->channels(); ++i) {
		channels[i] = this->get_channel(i);
	}

	return channels;
}

void image_array::merge( const vector<image_array> &channels) {
	ASSERT(channels.length() > 0 && channels.length() < 5);
	if (channels.length() == 1) {
		*this = channels[0].clone();
		return;
	}

	auto rows = channels[0].rows();
	auto cols = channels[0].cols();
	auto ch_count = channels.length();
	auto dtype = channels[0].dtype();

	for (auto c : channels) {
		ASSERT(c.rows() == rows && c.cols() == cols && c.channels() == 1 && c.dtype() == dtype);
	}

	this->create(rows, cols, ch_count, dtype);
	auto item_stride = this->strides()[2];

	for (unsigned i = 0; i < rows; ++i) {
		for (unsigned j = 0; j < cols; ++j) {
			for (unsigned k = 0; k < ch_count; ++k) {
				std::memcpy(
						reinterpret_cast<void*>(&this->at_index(i, j, k)), 
						reinterpret_cast<void*>(const_cast<unsigned char*>(&(channels[k].at_index(i, j, 0)))), 
						item_stride);
			}
		}
	}
}

image_array image_array::get_channel(unsigned channel) {
	ASSERT(*this && channel < this->channels());

	image_array c_i;

	c_i._data = this->_data;
	c_i._begin = this->_begin + channel*this->_strides[2];
	c_i._strides = {this->_strides[0], this->_strides[1], this->_strides[2]*this->channels()};
	c_i._shape = {this->rows(), this->cols(), 1};
	c_i._refcount = this->_refcount;
	c_i._dtype = this->_dtype;

	REF_INCREMENT(c_i._refcount);

	return c_i;
}

const image_array image_array::get_channel(unsigned channel) const {
	ASSERT(*this && channel < this->channels());

	image_array c_i;

	c_i._data = this->_data;
	c_i._begin = this->_begin + channel*this->_strides[2];
	c_i._strides = {this->_strides[0], this->_strides[1], this->_strides[2]*this->channels()};
	c_i._shape = {this->rows(), this->cols(), 1};
	c_i._refcount = this->_refcount;

	REF_INCREMENT(c_i._refcount);

	return c_i;
}

void image_array::to_gray() {
    ASSERT(this->is_valid());
    if (this->channels() == 1) {
        return;
    }
    image_array tgt_img(this->rows(), this->cols(), 1, this->dtype());
    if (this->channels() >= 3) {
        // TODO: support all other types here as well.
        switch (this->dtype()) {
        case UINT8:
            internal::cv_to_gray<byte>(*this, tgt_img);
            break;
        case UINT16:
            internal::cv_to_gray<unsigned short>(*this, tgt_img);
            break;
        case FLOAT32:
            internal::cv_to_gray<float>(*this, tgt_img);
            break;
        case FLOAT64:
            internal::cv_to_gray<double>(*this, tgt_img);
            break;
        default:
            throw std::runtime_error("Type convertion not supported");
        }
    } else {
        throw std::runtime_error("Invalid source image!~ image should be a RGB, or RGBA to be converted to gray");
    }
    *this = std::move(tgt_img);
}

void image_array::to_rgb() {
    ASSERT(this->is_valid());
    if (this->channels() == 3) {
        return;
    }
    image_array tgt_img(this->rows(), this->cols(), 3, this->dtype());
    switch (this->dtype()) {
    case UINT8:
        internal::cv_to_rgb<byte>(*this, tgt_img);
        break;
    case UINT16:
        internal::cv_to_rgb<unsigned short>(*this, tgt_img);
        break;
    case FLOAT32:
        internal::cv_to_rgb<float>(*this, tgt_img);
        break;
    case FLOAT64:
        internal::cv_to_rgb<double>(*this, tgt_img);
        break;
    default:
        throw std::runtime_error("Type not supported");
    }
    *this = std::move(tgt_img);
}


void image_array::to_rgba() {
    ASSERT(this->is_valid());
    if (this->channels() == 4) {
        return;
    }
    image_array tgt_img(this->rows(), this->cols(), 4, this->dtype());
    switch (this->dtype()) {
    case UINT8:
		internal::cv_to_rgba<byte>(*this, tgt_img);
        break;
    case UINT16:
		internal::cv_to_rgba<unsigned short>(*this, tgt_img);
        break;
    case FLOAT32:
		internal::cv_to_rgba<float>(*this, tgt_img);
        break;
    case FLOAT64:
		internal::cv_to_rgba<double>(*this, tgt_img);
        break;
    default:
        throw std::runtime_error("Type not supported");
    }
    *this = std::move(tgt_img);
}

}

