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
//
// Description:
// Contains array structure specified to represent an image byte array. It's interface 
// and implementation is modeled to be similar with OpenCV cv::Mat structure, for
// sake of familiarity in usage.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef IMAGE_HPP_7D0HB8PC
#define IMAGE_HPP_7D0HB8PC


#include "array.hpp"
#include "matrix.hpp"
#include "matfunc.hpp"


#define CHECK_IMAGE_DEPTH(depth) ASSERT(depth == 1 || depth == 2 || depth == 4 || depth == 8)


namespace cv {

enum data_type {
    NONE = 0,
    UINT8,
    INT8,
    UINT16,
    INT16,
    UINT32,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
#ifdef REAL_TYPE_DOUBLE
	REAL = FLOAT64
#else
	REAL = FLOAT32
#endif
};


namespace internal {

const unsigned data_size[] {0, 1, 1, 2, 2, 4, 4, 8, 4, 8};

template<class _Tp>
data_type get_dtype() {
    if (typeid(unsigned char) == typeid(_Tp)) {
        return UINT8;
    } else if (typeid(char) == typeid(_Tp)) {
        return  INT8;
    } else if (typeid(unsigned short) == typeid(_Tp)) {
        return  UINT16;
    } else if (typeid(short) == typeid(_Tp)) {
        return  INT16;
    } else if (typeid(unsigned) == typeid(_Tp)) {
        return  UINT32;
    } else if (typeid(int) == typeid(_Tp)) {
        return  INT32;
    } else if (typeid(long) == typeid(_Tp)) {
        return  INT64;
    } else if (typeid(float) == typeid(_Tp)) {
        return  FLOAT32;
    } else if (typeid(double) == typeid(_Tp)) {
        return FLOAT64;
    }
    return NONE;
}

}

/*!
 * @brief image_array buffer class.
 *
 * Holds image_array data of arbitrary format and type.
 * Designed to be used as an I/O structure.
 * For manipulation, convert to matrix of
 * corresponding type.
 */
class CV_EXPORT image_array: public basic_array <byte> {
public:
private:
    data_type _dtype;

public:
    typedef basic_array<byte> super_type;

    //! Runtime assignment of the data type.
    template <typename _Tp> void set_type();
    template <typename _Tp> data_type get_type() const;
public:
    //! Default constructor.
    image_array();
    //! Constructor with parameter input.
    image_array(unsigned rows, unsigned cols, unsigned channels, data_type dtype);
    //! Constructor with parameter and data input.
    template<typename _Tp>
    image_array(unsigned rows, unsigned cols, unsigned channels, _Tp *data, refcount_type *refcount = nullptr);
    //! Constructor using single channeled matrix.
    template<typename _Tp>
    image_array(const matrix<_Tp> &matrix, bool deep_copy = false);
    //! Constructor using multi channel matrix.
    template<typename _Tp, unsigned _Channels>
    image_array(const matrix<vectorx<_Tp, _Channels> > &matrix, bool deep_copy = false);
	//! Construct multi-channel image using array of single-channel images.
	image_array(const vector<image_array> &channels);
    //! Copy constructor.
    image_array(const image_array &cpy, bool deep_copy = false);
    //! Move constructor.
    image_array(image_array &&move);
    //! Class destructor.
    virtual ~image_array();

    //! Assignment operator.
    image_array &operator=(const image_array &rhs) = default;
    //! Move operator.
    image_array &operator=(image_array &&rhs);

    //! Constructor method with parameter input.
    void create(unsigned rows, unsigned cols, unsigned channels, data_type dtype);
    //! Constructor method with parameter and data input by referencing data.
    template<typename _Tp>
    void create(unsigned rows, unsigned cols, unsigned channels, _Tp *data);

    //! Get a clone of this image_array. Performs deep copy.
    image_array clone() const;

    template<typename _Tp> inline
    _Tp &at(unsigned row, unsigned col, unsigned channel = 0);
    template<typename _Tp> inline
    const _Tp &at(unsigned row, unsigned col, unsigned channel = 0) const;

    //! Release image array data.
    virtual void release();

    //! Get channel count.
    unsigned channels() const;
    //! Return rows count.
    unsigned rows() const;
    //! Return column count.
    unsigned cols() const;
    //! Get size of the matrix.
    vec3i size() const;
    //! Get data type of the image.
    data_type dtype() const;
    //! Get size of row in bytes (columns * channels * depth).
    unsigned row_stride() const;
    //! Get data depth.
    unsigned depth() const;

    /*!
     * @brief Split multichannel image_array to array of individual channels.
     *
     * Does not perform copy, but makes a reference array for each channel.
     */
    vector<image_array> split();

    /*!
     * @brief Construct the image array from given channels.
     *
     * This method creates new array then copies channel data in the
     * merged array.
     */
    void merge(const vector<image_array> &channels);

	/*!
	 * @brief Get reference to isolated channel of the image array.
	 *
	 * Complexity O(1).
	 */
	image_array get_channel(unsigned channel);

	/*!
	 * @brief Get read-only reference to isolated channel of the image array.
	 *
	 * Complexity O(1).
	 */
	const image_array get_channel(unsigned channel) const;

    //! Convert image_array to grayscale image_array.
    void to_gray();
    //! Convert grayscale image_array to 3-channel image_array.
    void to_rgb();
    //! Convert image_array to 4-channel image_array. Assumed current model
    void to_rgba();
	//! Convert data type by giving an enum representation of the data.
	inline void convert_to(data_type dtype, bool typeCheck = true);
	//! Return copy of this array with cast of the data - similar behaviour as numpy.astype.
	inline image_array as_type(data_type dtype) const;
    //! Convert buffer depth - data type.
    template <typename _Tp> void convert_to(bool typeCheck = true);

    //! Type trait to check the data type of the image data.
    template <typename _Tp> bool is_type() const;
    /*!
     * @brief Operator converter to corresponding matrix.
     *
     * This operator will performs reference assignment, so no deep copy is performed.
     */
    template<typename _Tp>
    operator matrix<_Tp>() const {
        ASSERT(this->is_valid() && sizeof(_Tp) == this->depth());
        return matrix<_Tp>(this->rows(), this->cols(), reinterpret_cast<_Tp*>(this->_begin), this->_refcount);
    }

    //! Boolean operator, checking for data validity.
    operator bool() const {
        return static_cast<bool>(this->_data);
    }
};

template<typename _Tp>
image_array::image_array(unsigned rows, unsigned cols, unsigned channels, _Tp *data, refcount_type *refcounter): super_type() {

    auto t_size = sizeof(_Tp);

    CHECK_IMAGE_DEPTH(t_size);
    ASSERT(rows && cols && channels && data);

    if (refcounter) {
        this->_refcount = refcounter;
        REF_INCREMENT(refcounter);
        this->_data = reinterpret_cast<byte*>(data);
        this->_begin = reinterpret_cast<byte*>(data);
        this->_shape = {rows, cols, channels};
        this->_strides = {static_cast<unsigned>(cols*t_size), static_cast<unsigned>(channels*t_size), static_cast<unsigned>(t_size)};
    } else {
        this->allocate({rows, cols, channels}, sizeof(_Tp));
        this->_refcount = REF_NEW;
        std::copy(data, data + rows*cols*channels, reinterpret_cast<_Tp*>(this->_begin));
    }
}

template<typename _Tp>
image_array::image_array(const matrix<_Tp> &matrix, bool deep_copy) : super_type() {

    this->set_type<_Tp>();

    if (!matrix)
        return;
    else {
        if (deep_copy) {
            this->allocate({matrix.rows(), matrix.cols(), 1}, sizeof(_Tp));
            this->_refcount = REF_NEW;
            if (matrix.is_contiguous()) {
                std::memcpy(reinterpret_cast<void*>(this->_data), &*matrix.begin(), matrix.shape().product()*sizeof(_Tp));
            } else {
                for (unsigned i = 0; i < this->rows(); ++i) {
                    for (unsigned j = 0; j < this->cols(); ++j) {
                        *(reinterpret_cast<_Tp*>(&(this->at_index(i, j, 0)))) = matrix(i, j);
                    }
                }
            }
        } else {
            this->_data = reinterpret_cast<pointer>(matrix.data());
            this->_begin = reinterpret_cast<pointer>(matrix.data_begin());
            this->_shape = {matrix.rows(), matrix.cols(), 1};
            this->_strides = {matrix.cols()*sizeof(_Tp), sizeof(_Tp), sizeof(_Tp)};
            this->_refcount = matrix.refcounter();
            REF_INCREMENT(this->_refcount);
        }
    }
}

template<typename _Tp, unsigned _Channels>
image_array::image_array(const matrix<vectorx<_Tp, _Channels> > &matrix, bool deep_copy) : super_type() {
    this->set_type<_Tp>();
    if (!matrix.is_valid()) {
        return;
    }
    if (deep_copy) {
        this->allocate(matrix.rows(), matrix.cols(), _Channels, sizeof(_Tp));
        this->_refcount = matrix._refcount;
        REF_INCREMENT(this->_refcount);
    } else {
        this->_data = reinterpret_cast<byte*>(matrix._data);
        this->_begin = reinterpret_cast<byte*>(matrix._begin);
        this->_shape = {matrix.rows(), matrix.cols(), _Channels};
        this->_strides = {matrix.cols()*_Channels*sizeof(_Tp), _Channels*sizeof(_Tp), sizeof(_Tp)};
        this->_refcount = matrix._refcount;
        REF_INCREMENT(this->_refcount);
    }
}

template<typename _Tp>
void image_array::create(unsigned rows, unsigned cols, unsigned channels, _Tp *data) {
    this->create(rows, cols, channels, internal::get_dtype<_Tp>());
    std::copy(data, data + rows*cols*channels, reinterpret_cast<_Tp*>(this->_begin));
}

template<typename _Tp>
_Tp &image_array::at(unsigned row, unsigned col, unsigned channel) {
    ASSERT(this->is_valid() && this->is_type<_Tp>() &&
           row < this->rows() && col < this->cols() &&
           channel < this->channels());
    return *reinterpret_cast<_Tp*>(&this->at_index(row, col, channel));
}

template<typename _Tp>
const _Tp &image_array::at(unsigned row, unsigned col, unsigned channel) const {
    ASSERT(this->is_valid() && this->is_type<_Tp>() &&
           row < this->rows() && col < this->cols() &&
           channel < this->channels());
    return *reinterpret_cast<_Tp*>(const_cast<byte*>(&this->at_index(row, col, channel)));
}

namespace internal {
template<typename _SourceType, typename _TargetType>
void _cv_convert_to(const image_array &source, image_array &target, bool typeCheck) {
    if (typeCheck) {
        for (unsigned i = 0; i < source.rows(); ++i) {
            for (unsigned j = 0; j < source.cols(); ++j) {
                for (unsigned c = 0; c < target.channels(); c++)
                    target.at<_TargetType>(i, j, c) = ranged_cast<_TargetType>(source.at<_SourceType>(i, j, c));
            }
        }
    } else {
        for (unsigned i = 0; i < source.rows(); ++i) {
            for (unsigned j = 0; j < source.cols(); ++j) {
                for (unsigned c = 0; c < target.channels(); c++)
                    target.at<_TargetType>(i, j, c) = source.at<_SourceType>(i, j, c);
            }
        }
    }
}
}

void image_array::convert_to(data_type dt, bool type_check) {
	ASSERT(*this && dt != NONE);
	if (dt == this->dtype())
		return;
    image_array tgt_img(this->rows(), this->cols(), this->channels(), dt);
    switch (this->dtype()) {
    case UINT8:
        switch (dt) {
        case UINT8:
            internal::_cv_convert_to<byte, byte>(*this, tgt_img, type_check);
            break;
        case UINT16:
            internal::_cv_convert_to<byte, unsigned short>(*this, tgt_img, type_check);
            break;
        case FLOAT32:
            internal::_cv_convert_to<byte, float>(*this, tgt_img, false);
            break;
        case FLOAT64:
            internal::_cv_convert_to<byte, double>(*this, tgt_img, false);
            break;
        default:
            std::runtime_error("Invalid target image array depth");
        }
        break;
    case UINT16:
        switch (dt) {
        case UINT8:
            internal::_cv_convert_to<unsigned short, byte>(*this, tgt_img, type_check);
            break;
        case UINT16:
            internal::_cv_convert_to<unsigned short, unsigned short>(*this, tgt_img, type_check);
            break;
        case FLOAT32:
            internal::_cv_convert_to<unsigned short, float>(*this, tgt_img, false);
            break;
        case FLOAT64:
            internal::_cv_convert_to<unsigned short, double>(*this, tgt_img, false);
            break;
        default:
            std::runtime_error("Invalid target image array depth");
        }
        break;
    case FLOAT32:
        switch (dt) {
        case UINT8:
            internal::_cv_convert_to<float, byte>(*this, tgt_img, type_check);
            break;
        case UINT16:
            internal::_cv_convert_to<float, unsigned short>(*this, tgt_img, type_check);
            break;
        case FLOAT32:
            internal::_cv_convert_to<float, float>(*this, tgt_img, false);
            break;
        case FLOAT64:
            internal::_cv_convert_to<float, double>(*this, tgt_img, false);
            break;
        default:
            std::runtime_error("Invalid target image array depth");
        }
        break;
    case FLOAT64:
        switch (dt) {
        case UINT8:
            internal::_cv_convert_to<double, byte>(*this, tgt_img, type_check);
            break;
        case UINT16:
            internal::_cv_convert_to<double, unsigned short>(*this, tgt_img, type_check);
            break;
        case FLOAT32:
            internal::_cv_convert_to<double, float>(*this, tgt_img, false);
            break;
        case FLOAT64:
            internal::_cv_convert_to<double, double>(*this, tgt_img, false);
            break;
        default:
            std::runtime_error("Invalid target image array depth");
        }
        break;
    default:
        std::runtime_error("Invalid source image array depth");
    }
    *this = std::move(tgt_img);
}

image_array image_array::as_type(data_type dtype) const {
	auto out = this->clone();
	out.convert_to(dtype);
	return out;
}

template<typename _Tp>
void image_array::convert_to(bool type_check) {
    // TODO: finis for all types!
    auto dt = internal::get_dtype<_Tp>();
    if (dt == NONE) {
        throw std::runtime_error("Invalid convertion type");
    }
	this->convert_to(dt, type_check);
}

template<class _Tp>
void image_array::set_type() {
    if (typeid(unsigned char) == typeid(_Tp)) {
        this->_dtype = UINT8;
    } else if (typeid(char) == typeid(_Tp)) {
        this->_dtype = INT8;
    } else if (typeid(unsigned short) == typeid(_Tp)) {
        this->_dtype = UINT16;
    } else if (typeid(short) == typeid(_Tp)) {
        this->_dtype = INT16;
    } else if (typeid(unsigned) == typeid(_Tp)) {
        this->_dtype = UINT32;
    } else if (typeid(int) == typeid(_Tp)) {
        this->_dtype = INT32;
    } else if (typeid(long) == typeid(_Tp)) {
        this->_dtype = INT64;
    } else if (typeid(float) == typeid(_Tp)) {
        this->_dtype = FLOAT32;
    } else if (typeid(double) == typeid(_Tp)) {
        this->_dtype = FLOAT64;
    } else {
        throw std::runtime_error("Invalid type for image_array");
    }
}

template<class _Tp>
bool image_array::is_type() const {
    switch(this->_dtype) {
    case UINT8:
        return (typeid(unsigned char) == typeid(_Tp));
    case INT8:
        return (typeid(char) == typeid(_Tp));
    case UINT16:
        return (typeid(unsigned short) == typeid(_Tp));
    case INT16:
        return (typeid(short) == typeid(_Tp));
    case UINT32:
        return (typeid(unsigned int) == typeid(_Tp));
    case INT32:
        return (typeid(int) == typeid(_Tp));
    case INT64:
        return (typeid(long) == typeid(_Tp));
    case FLOAT32:
        return (typeid(float) == typeid(_Tp));
    case FLOAT64:
        return (typeid(double) == typeid(_Tp));
    default:
        return false;
    }
}

}

#endif /* end of include guard: IMAGE_HPP_7D0HB8PC */
