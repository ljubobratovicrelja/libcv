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
//
// Description:
// MxN Matrix class implementation.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef MATRIX_HPP_AEUQVFPW
#define MATRIX_HPP_AEUQVFPW


#include "vector.hpp"


namespace cv {

/*!
 * @brief Image matrix class.
 *
 */
template<class _Tp>
class matrix: public basic_array < _Tp > {
  public:
	typedef CV_TYPENAME basic_array<_Tp>::value_type value_type;
	typedef CV_TYPENAME basic_array<_Tp>::pointer pointer;
	typedef CV_TYPENAME basic_array<_Tp>::reference reference;
	typedef CV_TYPENAME basic_array<_Tp>::const_reference const_reference;
	typedef CV_TYPENAME basic_array<_Tp>::difference_type difference_type;
	typedef CV_TYPENAME basic_array<_Tp>::size_type size_type;

	typedef bidirectional_iterator<_Tp> iterator; //!< bidirectional iterator with respect to strides.
	typedef bidirectional_iterator<const _Tp> const_iterator; //!< bidirectional read-only iterator with respect to strides.

	typedef basic_array<_Tp> super_type; //!< Type of the superclass.

  protected:
	//! Allocate region from another matrix.
	void copy_roi(unsigned row, unsigned col, unsigned rows, unsigned cols, const matrix<_Tp>& source, RoiEdge edgeManage);
	//! Reference region of data from another matrix.
	void ref_roi(unsigned row, unsigned col, unsigned rows, unsigned cols, const matrix<_Tp>& source);

  public:
	//! Default constructor.
	matrix();
	//! Class constructor.
	matrix(unsigned rows, unsigned cols);
	//! Matrix initializer constructor.
	matrix(const std::initializer_list<std::initializer_list<_Tp> > &m);
	//! Class constructor.
	matrix(vec2i size);

	//! Class constructor.
	matrix(unsigned rows, unsigned cols, pointer data, refcount_type *refcounter = nullptr);
	//! Class constructor.
	matrix(pointer data, pointer begin, const index_array &shape, const index_array &strides, refcount_type *refcounter);
	//! Copy constructor.
	matrix(const matrix<_Tp> &m, bool deep_copy = false);
	//! Constructor from cv::vector
	matrix(const vector<_Tp> &v, bool transposed = false, bool deep_copy = false);
	//! Constructor from 3D cv::vectorx
	matrix(const vectorx<_Tp, 3> &v, bool transposed = false);
	//! Constructor from 4D cv::vectorx
	matrix(const vectorx<_Tp, 4> &v);
	//! Move constructor.
	matrix(matrix<_Tp> &&m);

	//! Roi constructor
	matrix(unsigned row, unsigned col, unsigned rows, unsigned cols, const matrix<_Tp>& source, RoiType allocation = RoiType::COPY,
	       RoiEdge edgeManage = RoiEdge::MIRROR);

	//! Assignment operator.
	matrix<_Tp>& operator=(const matrix<_Tp> &m) = default;
	//! Assignment move operator.
	matrix<_Tp>& operator=(matrix<_Tp> &&m);

	//! Class destructor.
	virtual ~matrix();

	//! Create matrix with n cols, m cols.
	void create(unsigned rows, unsigned cols);
	//! Create matrix with n cols, m cols.
	void create(unsigned rows, unsigned cols, const _Tp & val);
	//! Create matrix with size.
	void create(vec2i size);
	//! Create matrix with size.
	void create(vec2i size, const _Tp &val);
	//! Copy another matrix's data.
	void create(const matrix<_Tp>& rhs);

	//! Create submatrix from other matrix.
	void create(unsigned row, unsigned col, unsigned rows, unsigned cols, const matrix<_Tp>& source, RoiType allocation =
	                RoiType::REFERENCE, RoiEdge edgeManage = RoiEdge::MIRROR);

	//! Matlab-like static zeros initializer. Creates matrix filled with zeros of appropriate type.
	static matrix<_Tp> zeros(unsigned rows, unsigned cols);
	//! Matlab-like static eye initializer. Creates unit matrix of given square size.
	static matrix<_Tp> eye(unsigned size);

	//! Matlab-like static zeros initializer. Creates matrix filled with zeros of appropriate type.
	static matrix<_Tp> zeros(vec2i size);

	//! Return rows count.
	unsigned rows() const;
	//! Return column count.
	unsigned cols() const;
	//! Get size of the matrix.
	vec2i size() const;

	/*!
	* @brief Get row at index.
	*
	* Data is reference, not a copy.
	* Complexity is O(1).
	*
	*/
	vector<_Tp> row(unsigned i);
	/*!
	* @brief Get column at index.
	*
	* Data is reference, not a copy.
	* Complexity is O(1).
	*
	*/
	vector<_Tp> col(unsigned i);

	/*!
	* @brief Fill all members with given value.
	*/
	void fill(const _Tp& value);
	//! Check if this matrix is square.
	bool is_square() const;

	//! Transpose matrix.
	matrix<_Tp> &transpose();
	//! Return transposed version, but do not alter this matrix data.
	matrix<_Tp> transposed() const;

	//! Get mutable data begin iterator.
	iterator begin();
	//! Get mutable data end iterator.
	iterator end();
	//! Get immutable data begin iterator.
	const_iterator begin() const; 
	//! Get immutable data begin iterator.
	const_iterator end() const;
	//!Swap two matrix item values.
	void swap(unsigned i1, unsigned i2);
	//!Swap two matrix item values.
	void swap(unsigned r1, unsigned c1, unsigned r2, unsigned c2);
	//! Swap two matrix row values.
	void swapRows(unsigned r1, unsigned r2);
	//! Swap two matrix column values.
	void swapCols(unsigned c1, unsigned c2);

	//! Get copy (clone) of this matrix.
	matrix<_Tp> clone() const;
	//! Zero out matrix - assign zero values to all items of the matrix.
	void to_zero();
	//! Reset this matrix to identity. Throws exception if matrix is not square.
	void to_identity();
	//! Reshape matrix. Change row and column size. For now works only on contiguous matrices.
	void reshape(unsigned rows, unsigned cols);

	//! Index operator.
	_Tp& operator()(unsigned row, unsigned col);
	//! Index operator.
	const _Tp& operator()(unsigned row, unsigned col) const;

	//! Comparison operator.
	virtual bool operator==(const matrix<_Tp>& m);
	//! Comparison operator.
	virtual bool operator!=(const matrix<_Tp>& m);

	//! Cast to a vector.
	operator vector<_Tp>();

	//! matrix data validity check.
	operator bool() const;
	//! Cast matrix to other typed matrix.
	template<class _Up>	operator matrix<_Up>() const;
};

typedef matrix<int> matrixi; //!< 32-bit int matrix
typedef matrix<unsigned> matrixui; //!< 32-bit unsigned int matrix
typedef matrix<float> matrixf; //!< 32-bit float matrix.
typedef matrix<double> matrixd; //!< 64-bit float matrix.
typedef matrix<unsigned char> matrixb; //!< 8-bit unsigned char matrix (byte).
typedef matrix<short> matrixs; //!< 16-bit short matrix.

typedef matrix<vec2d> matrix2d; //!< Two channel 64-bit double matrix
typedef matrix<vec2i> matrix2i; //!< Two channel 32-bit int matrix
typedef matrix<vec2f> matrix2f; //!< Two channel 32-bit float matrix
typedef matrix<vec2s> matrix2s; //!< Two channel 16-bit short matrix
typedef matrix<vec2b> matrix2b; //!< Two channel 8-bit byte matrix

typedef matrix<vec3d> matrix3d; //!< Three channel 64-bit double matrix
typedef matrix<vec3i> matrix3i; //!< Three channel 32-bit int matrix
typedef matrix<vec3f> matrix3f; //!< Three channel 32-bit float matrix
typedef matrix<vec3s> matrix3s; //!< Three channel 16-bit short matrix
typedef matrix<vec3b> matrix3b; //!< Three channel 8-bit byte matrix

typedef matrix<vec4d> matrix4d; //!< Four channel 64-bit double matrix
typedef matrix<vec4i> matrix4i; //!< Four channel 32-bit int matrix
typedef matrix<vec4f> matrix4f; //!< Four channel 32-bit float matrix
typedef matrix<vec4s> matrix4s; //!< Four channel 16-bit short matrix
typedef matrix<vec4b> matrix4b; //!< Four channel 8-bit byte matrix

#ifdef CV_REAL_TYPE_DOUBLE
typedef matrixd matrixr;
typedef matrix2d matrix2r;
typedef matrix3d matrix3r;
typedef matrix4d matrix4r;
#else
typedef matrixf matrixr;
typedef matrix2f matrix2r;
typedef matrix3f matrix3r;
typedef matrix4f matrix4r;
#endif

// Implementation

template<typename _Tp>
matrix<_Tp>::matrix():
	super_type() {
}

template<typename _Tp>
matrix<_Tp>::matrix(unsigned rows, unsigned cols) :
	super_type(rows, cols) {
	ASSERT(rows && cols);
}

template<typename _Tp>
matrix<_Tp>::matrix(const std::initializer_list<std::initializer_list<_Tp> > &m) : super_type() {
	ASSERT(m.size() && m.begin()->size());
	unsigned r = m.size();
	unsigned c = m.begin()->size();
	this->allocate({r, c});
	auto i_iter = m.begin();
	for (unsigned i = 0; i < m.size(); ++i, ++i_iter) {
		auto iter = i_iter->begin();
		for (unsigned j = 0; j < i_iter->size(); ++j, ++iter) {
			ASSERT(i_iter->size() == c);
			(*this)(i, j) = *iter;
		}
	}
}

template<typename _Tp>
matrix<_Tp>::matrix(vec2i size):
	super_type(size[0], size[1]) {
}

template<typename _Tp>
matrix<_Tp>::matrix(unsigned rows, unsigned cols, pointer data, refcount_type *refcounter) : 
	super_type(data, data, {rows, cols}, {cols, 1}, refcounter) {
	ASSERT(rows && cols && data);
}

template<typename _Tp>
matrix<_Tp>::matrix(pointer data, pointer begin, const index_array &shape, const index_array &strides, refcount_type *refcount) : 
	super_type(data, begin, shape, strides, refcount) {
	ASSERT(data && begin && shape.size() == 2 && strides.size() == 2);
}

template<typename _Tp>
matrix<_Tp>::matrix(const matrix<_Tp> &m, bool deep_copy):
	super_type(m, deep_copy) {
}

template<typename _Tp>
matrix<_Tp>::matrix(const vector<_Tp> &v, bool transposed, bool deep_copy):
	super_type() {
		if (deep_copy) {
			this->_refcount = REF_NEW;

			if (transposed)
				this->allocate({v.length(), 1});
			else
				this->allocate({1, v.length()});

			std::copy(v.begin(), v.end(), this->_begin);
		} else {
			this->_data = v.data();
			this->_begin = v.data_begin();
			this->_refcount = v.refcounter();
			REF_INCREMENT(this->_refcount);
			if (transposed) {
				this->_shape = { v.length(), 1};
			} else {
				this->_shape = {1, v.length()};
			}
			this->_strides = {v.strides()[0]*v.length(), v.strides()[0]};
		}
}

template<typename _Tp>
matrix<_Tp>::matrix(const vectorx<_Tp, 3> &v, bool transposed): super_type() {
	this->_refcount = REF_NEW;
	if (transposed)
		this->allocate({3, 1});
	else
		this->allocate({1, 3});
	std::copy(v.begin(), v.end(), this->_data);
}

template<typename _Tp>
matrix<_Tp>::matrix(const vectorx<_Tp, 4> &v): super_type() {
	this->_refcount = REF_NEW;
	this->allocate({1, 4});
	std::copy(v.begin(), v.end(), this->_data);
}

template<typename _Tp>
matrix<_Tp>::matrix(matrix<_Tp> &&m) : super_type(m) {
}

template<typename _Tp>
matrix<_Tp>::matrix(unsigned row, unsigned col, unsigned rows, unsigned cols,
                    const matrix<_Tp>& source, RoiType allocation, RoiEdge edgeManage) {
	this->create(row, col, rows, cols, source, allocation, edgeManage);
}

template<typename _Tp>
matrix<_Tp>::~matrix() {
}

template<typename _Tp>
matrix<_Tp>& matrix<_Tp>::operator=(matrix<_Tp> &&m) {
	if (this != &m) {
		super_type::operator=(std::move(m));
	}
	return *this;
}

template<typename _Tp>
void matrix<_Tp>::create(unsigned rows, unsigned cols) {
	this->release();
	this->_refcount = REF_NEW;
	this->allocate({rows, cols});
}

template<typename _Tp>
void matrix<_Tp>::create(unsigned rows, unsigned cols, const _Tp & val) {
	this->create(rows, cols);
	this->fill(val);
}

template<typename _Tp>
void matrix<_Tp>::create(vec2i size) {
	this->create(size[0], size[1]);
}

template<typename _Tp>
void matrix<_Tp>::create(vec2i size, const _Tp & val) {
	this->create(size);
	this->fill(val);
}

template<typename _Tp>
void matrix<_Tp>::create(const matrix<_Tp>& cpy) {
	this->copy(cpy);
}

template<typename _Tp>
void matrix<_Tp>::create(unsigned row, unsigned col, unsigned rows, unsigned cols,
                         const matrix<_Tp>& source, RoiType allocation, RoiEdge edgeManage) {
	switch(allocation) {
	case RoiType::COPY:
		copy_roi(row, col, rows, cols, source, edgeManage);
		break;
	case RoiType::REFERENCE:
		ref_roi(row, col, rows, cols, source);
	}
}

template<typename _Tp>
matrix<_Tp> matrix<_Tp>::zeros(unsigned rows, unsigned cols) {
	matrix<_Tp> z(rows, cols);
	z.fill(0);
	return z;
}

template<typename _Tp>
matrix<_Tp> matrix<_Tp>::eye(unsigned size) {
	matrix<_Tp> eye_mat(size, size);
	for (unsigned i = 0; i < size; ++i) {
		for (unsigned j = 0; j < size; ++j) {
			if (i == j) {
				eye_mat(i, j) = static_cast<_Tp>(1);
			} else {
				eye_mat(i, j) = static_cast<_Tp>(0);
			}
		}
	}
	return eye_mat;
}

template<typename _Tp>
matrix<_Tp> matrix<_Tp>::zeros(vec2i size) {
	matrix<_Tp> z(size);
	z.fill(0);
	return z;
}

template<typename _Tp>
void matrix<_Tp>::copy_roi(unsigned row, unsigned col, unsigned rows, unsigned cols,
                           const matrix<_Tp>& source, RoiEdge edgeManage) {

	ASSERT(
	    source && (row + rows / 2) >= 0 && (col + cols / 2) >= 0
	    && (row - rows / 2) < (int)source.rows()
	    && (col - cols / 2) < (int)source.cols());

	if (this->rows() != rows || this->cols() != cols) {
		// reallocate
		this->create(rows, cols);
	}

	if (edgeManage == RoiEdge::MIRROR) {
		for (unsigned i = 0; i < rows; ++i) {
			for (unsigned j = 0; j < cols; ++j) {

				int roi_row = (
				                  (i + row < 0) ?
				                  (i + row) :
				                  (i + row >= source.rows() ?
				                   (source.rows() - 1)
				                   - (i + row - source.rows()) :
				                   i + row));
				int roi_col = (
				                  (j + col < 0) ?
				                  (j + col) :
				                  (j + col >= source.cols() ?
				                   (source.cols() - 1)
				                   - (j + col - source.cols()) :
				                   j + col));

				this->at_index(i, j) = source(roi_row, roi_col);
			}
		}
	} else if (edgeManage == RoiEdge::ZEROS) {
		for (unsigned i = 0; i < rows; ++i) {
			for (unsigned j = 0; j < cols; ++j) {
				int roi_row = i + row;
				int roi_col = j + col;

				if (roi_row >= 0 && roi_row < source.rows() && roi_col >= 0
				        && roi_col < source.cols()) {
					this->at_index(i, j) = source(roi_row, roi_col);
				} else {
					this->at_index(i, j) = _Tp(0);
				}
			}
		}
	}
}

template<typename _Tp>
void matrix<_Tp>::ref_roi(unsigned row, unsigned col, unsigned rows, unsigned cols, const matrix<_Tp>& source) {

	ASSERT(
	    source && row + rows < source.rows() && row >= 0 && col >= 0
	    && col + cols < source.cols());

	this->deallocate();

	this->_refcount = source._refcount;
	REF_INCREMENT(this->_refcount);

	this->_data = source._data;
	this->_begin = source._begin + row*source._strides[0] + col*source._strides[1];

	this->_shape = {rows, cols};
	this->_strides = source._strides;
}

// public methods /////////////////////////////////////////////////////

template<typename _Tp>
unsigned matrix<_Tp>::rows() const {
	return this->_shape[0];
}

template<typename _Tp>
unsigned matrix<_Tp>::cols() const {
	return this->_shape[1];
}

template<typename _Tp>
void matrix<_Tp>::fill(const _Tp& value) {
	for (unsigned i = 0; i < this->rows(); ++i) {
		for (unsigned j = 0; j < this->cols(); ++j) {
			this->at_index(i, j) = value;
		}
	}
}

template<typename _Tp>
bool matrix<_Tp>::is_square() const {
	return (this->rows() == this->cols() && this->_begin);
}

template<typename _Tp>
void matrix<_Tp>::reshape(unsigned rows, unsigned cols) {
	if (!*this) {
		return;
	}
	ASSERT(this->rows() * this->cols() == rows * cols && this->is_contiguous());
	this->_shape = {rows, cols};
	this->_strides = {cols, 1};
}

template<typename _Tp>
matrix<_Tp> &matrix<_Tp>::transpose() {
	if (!this->_data)
		return *this;

	matrix<_Tp> old = this->clone();
	this->reshape(this->cols(), this->rows());

	for (unsigned i = 0; i < this->rows(); ++i) {
		for (unsigned j = 0; j < this->cols(); ++j) {
			this->at_index(i, j) = old(j, i);
		}
	}

	return *this;
}

template<typename _Tp>
matrix<_Tp> matrix<_Tp>::transposed() const {
	if (!(*this))
		return matrix<_Tp>();
	matrix<_Tp> t(this->cols(), this->rows());
	for (unsigned i = 0; i < this->rows(); ++i) {
		for (unsigned j = 0; j < this->cols(); ++j) {
			t(j, i) = this->at_index(i, j);
		}
	}
	return t;
}

template<typename _Tp>
CV_TYPENAME matrix<_Tp>::iterator matrix<_Tp>::begin() {
	if (!this->_data)
		return nullptr;
	ASSERT(this->_strides[1]*this->cols() == this->_strides[0]);
	return iterator(this->_begin, this->_strides[1]);
}

template<typename _Tp>
CV_TYPENAME matrix<_Tp>::iterator matrix<_Tp>::end() {
	if (!this->_data)
		return nullptr;
	ASSERT(this->_strides[1]*this->cols() == this->_strides[0]);
	return iterator(this->_begin + this->_strides[0]*this->rows(), this->_strides[1]);
}

template<typename _Tp>
CV_TYPENAME matrix<_Tp>::const_iterator matrix<_Tp>::begin() const {
	if (!this->_data)
		return nullptr;
	ASSERT(this->_strides[1]*this->cols() == this->_strides[0]);
	return const_iterator(this->_begin, this->_strides[1]);
}

template<typename _Tp>
CV_TYPENAME matrix<_Tp>::const_iterator matrix<_Tp>::end() const {
	if (!this->_data)
		return nullptr;
	ASSERT(this->_strides[1]*this->cols() == this->_strides[0]);
	return const_iterator(this->_begin + this->_strides[0]*this->rows(), this->_strides[1]);
}

template<typename _Tp>
void matrix<_Tp>::swap(unsigned r1, unsigned c1, unsigned r2, unsigned c2) {
	ASSERT(r1 < this->rows() && c1 < this->cols() && r2 < this->rows() && c2 < this->cols());
	value_type temp = this->get(r1, c1);
	this->at_index(r1, c1) = this->at_index(r2, c2);
	this->at_index(r2, c2) = temp;
}

template<typename _Tp>
void matrix<_Tp>::swapRows(unsigned r1, unsigned r2) {
	ASSERT(r1 < rows() && r2 < rows());
	_Tp t;
	for (unsigned i = 0; i < this->cols(); ++i) {
		t = this->at_index(r1, i);
		this->at_index(r1, i) = this->at_index(r2, i);
		this->at_index(r2, i) = t;
	}
}

template<typename _Tp>
void matrix<_Tp>::swapCols(unsigned c1, unsigned c2) {
	ASSERT(c1 < cols() && c2 < cols());
	_Tp t;
	for (unsigned i = 0; i < this->rows(); ++i) {
		t = this->at_index(i,c1);
		this->at_index(i,c1) = this->at_index(i,c2);
		this->at_index(i,c2) = t;
	}
}

template<typename _Tp>
vec2i matrix<_Tp>::size() const {
	return vec2i(this->rows(), this->cols());
}

template<typename _Tp>
vector<_Tp> matrix<_Tp>::row(unsigned i) {
	ASSERT(i < this->rows());
	return vector<_Tp>(this->_data, this->_begin + i*this->_strides[0], this->cols(), this->_strides[1], this->_refcount);
}

template<typename _Tp>
vector<_Tp> matrix<_Tp>::col(unsigned i) {
	ASSERT(i < this->cols());
	return vector<_Tp>(this->_data, this->_begin + i*this->_strides[1], this->rows(), this->_strides[0], this->_refcount);
}

template<typename _Tp>
matrix<_Tp> matrix<_Tp>::clone() const {
	return matrix<_Tp>(*this, true);
}

template<typename _Tp>
void matrix<_Tp>::to_zero() {
	if (this->is_contiguous())
		std::memset(this->_begin, 0, this->_shape.product()*sizeof(_Tp));
	else {
		for (unsigned i = 0; i < this->rows(); ++i) {
			for (unsigned j = 0; j < this->cols(); ++j) {
				this->at_index(i, j) = _Tp(0);
			}
		}
	}
}

template<typename _Tp>
void matrix<_Tp>::to_identity() {
	ASSERT(*this && this->is_square());
	for (unsigned i = 0; i < this->rows(); ++i) {
		this->at_index(i, i) = static_cast<_Tp>(1);
	}
}

template<typename _Tp>
_Tp& matrix<_Tp>::operator()(unsigned row, unsigned col) {
	return this->at_index(row, col);
}

template<typename _Tp>
const _Tp& matrix<_Tp>::operator()(unsigned row, unsigned col) const {
	return this->at_index(row, col);
}

template<typename _Tp>
bool matrix<_Tp>::operator==(const matrix<_Tp>& m) {
	return this->_data == m._data;
}

template<typename _Tp>
bool matrix<_Tp>::operator!=(const matrix<_Tp>& m) {
	return !operator=(m);
}

template<typename _Tp>
matrix<_Tp>::operator bool() const {
	return static_cast<bool>(this->_data);
}

//! Cast matrix to other typed matrix.
template<typename _Tp>
template<typename _Up>
matrix<_Tp>::operator matrix<_Up>() const {
	if (!*this) {
		return matrix<_Up>();
	}

	matrix<_Up> new_type(this->size());

	NEST_FOR_TO(this->rows(), this->cols()) {
		new_type(i, j) = static_cast<_Up>(this->at_index(i, j));
	}

	return new_type;
}

template<typename _Tp>
std::ostream& operator<<(std::ostream& stream, const matrix<_Tp>& mat) {
	auto p = stream.precision();
	stream << std::fixed;
	stream.precision(5);
	for (int i = 0; i < mat.rows(); i++) {
		for (int j = 0; j < mat.cols(); j++) {
			stream << mat(i, j) << " ";
		}
		stream << std::endl;
	}
	stream.precision(p);
	return stream;
}

//-----------------------------------------------------------------------------------------
// matrix operator implementation
//-----------------------------------------------------------------------------------------

//! Addition operator.
template<typename _Tp>
matrix<_Tp> operator+(const matrix<_Tp> &lhs, const matrix<_Tp> &rhs) {

	ASSERT(
	    (lhs.rows() != 0 && lhs.cols() != 0)
	    && (lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols()));

	matrix<_Tp> retval(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retval(i, j) = lhs(i, j) + rhs(i, j);
	}

	return retval;
}

//! Subtraction operator.
template<typename _Tp>
matrix<_Tp> operator-(const matrix<_Tp>& lhs, const matrix<_Tp>& rhs) {

	ASSERT(
	    (lhs.rows() != 0 && lhs.cols() != 0)
	    && (lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols()));

	matrix<_Tp> retVal(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retVal(i, j) = lhs(i, j) - rhs(i, j);
	}

	return retVal;
}

//! matrix multiplication operator.
template<typename _Tp>
matrix<_Tp> operator*(const matrix<_Tp>& lhs, const matrix<_Tp>& rhs) {

	ASSERT(lhs && lhs.cols() == rhs.rows());

	matrix<_Tp> retVal(lhs.rows(), rhs.cols());

	NEST_FOR_TO(lhs.rows(), rhs.cols()) {

		retVal(i, j) = 0;

		for (int c = 0; c < rhs.rows(); c++) {
			retVal(i, j) += lhs(i, c) * rhs(c, j);
		}
	}
	return retVal;
}


//! Addition operator.
template<typename _Tp>
matrix<_Tp>& operator+=(matrix<_Tp>& lhs, const matrix<_Tp>& rhs) {
	lhs = lhs + rhs;
	return lhs;
}

//! Subtraction operator.
template<typename _Tp>
matrix<_Tp>& operator-=(matrix<_Tp>& lhs, const matrix<_Tp>& rhs) {
	lhs = lhs - rhs;
	return lhs;
}

//! Addition operator.
template<typename _Tp>
matrix<_Tp> operator+(const matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs.rows() != 0 && lhs.cols() != 0);

	matrix<_Tp> retval(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retval(i, j) = lhs(i, j) + rhs;
	}

	return retval;
}

//! Addition operator.
template<typename _Tp>
matrix<_Tp> operator-(const matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs.rows() != 0 && lhs.cols() != 0);

	matrix<_Tp> retval(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retval(i, j) = lhs(i, j) - rhs;
	}

	return retval;
}

//! Addition operator.
template<typename _Tp>
matrix<_Tp> operator/(const matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs.rows() != 0 && lhs.cols() != 0);

	matrix<_Tp> retval(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retval(i, j) = lhs(i, j) / rhs;

	}

	return retval;
}

//! Addition operator.
template<typename _Tp>
matrix<_Tp> operator*(const matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs.rows() != 0 && lhs.cols() != 0);

	matrix<_Tp> retval(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retval(i, j) = lhs(i, j) * rhs;

	}

	return retval;
}

//! Addition operator.
template<typename _Tp>
matrix<_Tp> operator%(const matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs.rows() != 0 && lhs.cols() != 0);

	matrix<_Tp> retval(lhs.rows(), lhs.cols());

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		retval(i, j) = (int)lhs(i, j) % (int)rhs;

	}

	return retval;
}

template<typename _Tp>
matrix<_Tp> &operator+=(matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs);

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		lhs(i, j) += rhs;
	}

	return lhs;
}

template<typename _Tp>
matrix<_Tp> &operator-=(matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs);

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		lhs(i, j) -= rhs;
	}

	return lhs;
}

template<typename _Tp>
matrix<_Tp> &operator/=(matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs);

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		lhs(i, j) /= rhs;
	}

	return lhs;
}

template<typename _Tp>
matrix<_Tp> &operator*=(matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs);

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		lhs(i, j) *= rhs;
	}

	return lhs;
}

template<typename _Tp>
matrix<_Tp> &operator%=(matrix<_Tp> &lhs, double rhs) {

	ASSERT(lhs);

	NEST_PARALLEL_FOR_TO(lhs.rows(), lhs.cols()) {
		lhs(i, j) = (int)lhs(i, j) % (int)rhs;
	}

	return lhs;
}

}

#endif /* end of include guard: MATRIX_HPP_AEUQVFPW */



