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
// Vector structure implementation.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef VECTOR_HPP_TRD8NTPW
#define VECTOR_HPP_TRD8NTPW


#include "array.hpp"


namespace cv {

namespace internal {

//! TODO: Figure out if range funcs should be included here, and this
// should be removed.
template<class const_iterator>
real_t norm(const_iterator begin, const_iterator end, Norm n) {
	real_t nv = 0;
	switch (n) {
	case Norm::L1:
		do {
			nv += std::fabs(*begin);
		} while (++begin != end);
		break;
	case Norm::L2:
		do {
			nv += *begin*(*begin);
		} while (++begin != end);
		nv = sqrt(nv);
		break;
	default:
		throw std::runtime_error("Norm type not supported");
	}
	return nv;
}

template<class iterator>
void normalize(iterator begin, iterator end, Norm n) {
	auto nv = internal::norm(begin, end, n);
	do {
		*begin /= nv;
	} while (++begin != end);
}
}

/*!
 * @brief Vector class with data allocated on heap.
 *
 */
template<class _Tp>
class vector: public basic_array < _Tp > {
  public:
	typedef _Tp value_type;
	typedef _Tp &reference;
	typedef const _Tp &const_reference;
	typedef _Tp *pointer;
	typedef const _Tp *const_pointer;
	typedef typename basic_array<_Tp>::size_type size_type;
	typedef typename basic_array<_Tp>::difference_type difference_type;

	typedef bidirectional_iterator<_Tp> iterator; //!< Iterator type.
	typedef bidirectional_iterator<const _Tp> const_iterator; //!< Read-only bidirectional_iterator type.

	typedef basic_array<_Tp> super_type; //! Type of the super class.

  public:
	//! Class constructor.
	vector();
	//! Construct vector from another range.
	vector(pointer data, pointer begin, unsigned length, unsigned stride = 1, refcount_type *refcount = nullptr);
	//! Construct vector from strange range.
	vector(pointer data, unsigned length, unsigned stride = 1, bool borrow = false);
	//! Construct vector of given size.
	vector(unsigned size);
	//! Initialize vector from the initializer list.
	vector(const std::initializer_list<_Tp> &list);
	//! Copy constructor.
	vector(const vector &cpy, bool deepCopy = false);
	//! Move constructor.
	vector(vector &&move);
	//! Class destructor.
	virtual ~vector();

	//! Assignment operator.
	vector &operator=(const vector &rhs) = default;
	//! Move operator.
	vector &operator=(vector &&rhs);

	//! Re-create the array with given size.
	void create(unsigned size);

	/*!
	 * @brief Create from another vector.
	 *
	 * @param from Which vector to copy from.
	 * @param deepCopy Should the data be referenced(false) or cloned(true)?
	 */
	void create(const vector &from, bool deepCopy = false);

	/*!
	 * Clone vector data.
	 * @return Cloned vector.
	 */
	vector clone() const;

	/*!
	 * Cast to other typed vector.
	 *
	 * Copies data, casts it, and
	 * returns as new typed vector.
	 */
	template<class _Up>	operator vector<_Up>() const;

	/*!
	 * @brief Get sub-vector at given indices.
	 *
	 * @param start Start index, where the sub-vector data starts.
	 * @param end End index.
	 * @param stride Stride of the sub-vector.
	 *
	 * @code
	 * cv::vector<int> vec = {1, 3, 4, 5};
	 * auto sub_vec = vec(0, 3, 2);
	 *
	 * ASSERT(sub_vec.length() == 2); // sub-vector is of length 2
	 * ASSERT(sub_vec[0] == vec[0] && sub_vec[1] == vec[2]); // sub-vector has every second item to the vector end.
	 * ASSERT(&sub_vec[0] == &vec[0] && &sub_vec[1] == &vec[2]); // sub-vector data is the same as the vector.
	 * @endcode
	 */
	vector<_Tp> operator()(unsigned start, unsigned end, unsigned stride = 1);

	//! Read-only sub-vector operator.
	const vector<_Tp> operator()(unsigned start, unsigned end, unsigned stride = 1) const;

	/*!
	 * @brief Iterator with begin of the data.
	 */
	iterator begin() {
		return iterator(this->_begin, this->_strides[0]);
	}
	/*!
	 * @brief Iterator with end of the data.
	 */
	iterator end() {
		return iterator(this->_begin + this->length()*this->_strides[0], this->_strides[0]);
	}
	//! Get read-only iterator for begin of the data.
	const_iterator begin() const {
		return const_iterator(reinterpret_cast<pointer>(this->_begin));
	}
	//! Get read-only iterator for begin of the data.
	const_iterator end() const {
		return const_iterator(reinterpret_cast<pointer>(this->_begin + this->length()*this->_strides[0]));
	}

	/*!
	 * Zero vector initializer.
	 *
	 * @param count Size of zero vector.
	 * @return Zero vector of size 'count'
	 */
	static vector<_Tp> zeros(unsigned count);

	/*!
	 * Ones vector initializer.
	 *
	 * @param count Size of ones vector.
	 * @return Ones vector of size 'count'
	 */
	static vector<_Tp> ones(unsigned count);

	//! Get the item with minimal value from the vector.
	iterator min();
	//! Get maximum value from the vector.
	iterator max();
	//! Get the item with minimal value from the vector.
	const_iterator min() const;
	//! Get maximum value from the vector.
	const_iterator max() const;

	//! Fill vector with given value.
	void fill(const _Tp &value);

	//! Calculate dot product.
	_Tp dot(const vector<_Tp> &rhs) const;
	//! Calculate cross product for 2 and 3 dimensional vector.
	vector<_Tp> cross(const vector<_Tp> &rhs) const;

	//! Calculate norm of the vector.
	real_t norm(Norm n = Norm::L2) const;
	//! Calculate distance of two vectors. Need to be of same size.
	real_t distance(const vector &rhs, Norm n = Norm::L2) const;

	//! Normalize this vector.
	void normalize(Norm n = Norm::L2);
	//! Get normalized copy of this vector.
	vector<_Tp> normalized(Norm n = Norm::L2) const;

	//! Sort array using default less ("<") operator.
	void sort();
	//! Sort array using custom comparison operator.
	template<class _Cmp> void sort(_Cmp cmp);

	//! Index operator.
	reference operator [](int index);
	//! Read-only index operator.
	const_reference operator [](int index) const;

	//! Addition operator.
	vector<_Tp> operator+(const vector<_Tp> &rhs) const;
	//! Remove operator.
	vector<_Tp> operator-(const vector<_Tp> &rhs) const;
	//! Multiply operator.
	vector<_Tp> operator*(const vector<_Tp> &rhs) const;
	//! Divide operator.
	vector<_Tp> operator/(const vector<_Tp> &rhs) const;
	//! Increment by value.
	vector<_Tp> &operator+=(const vector<_Tp> &rhs);
	//! Decrement by value.
	vector<_Tp> &operator-=(const vector<_Tp> &rhs);
	//! Multiply with value.
	vector<_Tp> &operator*=(const vector<_Tp> &rhs);
	//! Divide with value.
	vector<_Tp> &operator/=(const vector<_Tp> &rhs);

	vector<_Tp> operator+(const _Tp &rhs) const;
	vector<_Tp> operator-(const _Tp &rhs) const;
	vector<_Tp> operator*(const _Tp &rhs) const;
	vector<_Tp> operator/(const _Tp &rhs) const;
	//! Increment by value.
	vector<_Tp> &operator+=(const _Tp &rhs);
	//! Decrement by value.
	vector<_Tp> &operator-=(const _Tp &rhs);
	//! Multiply with value.
	vector<_Tp> &operator*=(const _Tp &rhs);
	//! Divide with value.
	vector<_Tp> &operator/=(const _Tp &rhs);
	bool operator ==(const vector<_Tp> &rhs) const;
	bool operator !=(const vector<_Tp> &rhs) const;
	bool operator <(const vector<_Tp> &rhs) const;
	//! vector data validity check.
	operator bool() const;
	//! Stream operator.
	friend std::ostream& operator<<(std::ostream& stream, const vector<_Tp> &arr) {
		if (arr.empty())
			return stream;
		stream << arr[0];
		for (unsigned i = 1; i < arr.length(); i++) {
			stream << " " << arr[i];
		}
		return stream;
	}
};

/**
 * @brief Fixed size vector structure.
 *
 * Designed to be used as n length vector
 * with arbitrary typed data with statically defined length.
 * One of main purposes for this class is
 * to define channels in templated matrix.
 * (e.g. Matrix<Vec3b> for 3 channels byte matrix.)
 */
template<class _Tp, unsigned _size>
class vectorx {
	static_assert(_size > 0, "Size of the vector has to be at least 1");
  public:
	typedef random_access_iterator<_Tp> iterator;  //!< random access iterator.
	typedef random_access_iterator<const _Tp> const_iterator; //!< read-only random access iterator.

  protected:
	_Tp _data[_size];  //!< Data of the vector.

  public:
	//! Default constructor.
	vectorx();
	//! Construct using initializer list of values.
	vectorx(const std::initializer_list<_Tp> &list);
	//! Constructor using scalar value - fills vector with it.
	vectorx(const _Tp &a);
	//! Vector-2 initializer. Vector needs to be previously defined as length 2.
	vectorx(const _Tp & a, const _Tp & b);
	//! Vector-3 initializer. Vector needs to be previously defined as length 3.
	vectorx(const _Tp & a, const _Tp & b, const _Tp & c);
	//! Vector-4 initializer. Vector needs to be previously defined as length 4.
	vectorx(const _Tp & a, const _Tp & b, const _Tp & c, const _Tp & d);
	//! Copy constructor.
	vectorx(const vectorx<_Tp, _size> &cpy);
	//! Merging constructor - merges two half-sized vectors to one.
	vectorx(const vectorx<_Tp, _size / 2> &rhs, const vectorx<_Tp, _size / 2> &lhs);
	//! Class destructor.
	~vectorx();

	//! Get pointer to the beginning of the vector data.
	_Tp *data();

	//! Get item size(count) of the vector.
	unsigned size() const;

	//! Get byte size of the vector. It is defined as item count * sizeof(value_type)
	unsigned byte_size() const;

	//! Assignment operator to scalar value - same as construction of a scalar value.
	template<class _Up>
	vectorx<_Tp, _size> operator=(const _Up & rhs);

	//! Assignment operator.
	vectorx<_Tp, _size> operator=(const vectorx<_Tp, _size> &rhs);

	//! Assignment operator to different sized vector. Fills the vector with overlapping indexed item values.
	template<unsigned o_size>
	vectorx<_Tp, _size> operator=(const vectorx<_Tp, o_size> &rhs);

	//! Addition operator.
	vectorx<_Tp, _size> operator+(const vectorx<_Tp, _size>& rhs) const;
	//! Increment operator.
	vectorx<_Tp, _size>& operator++();
	//! Item-wise addition.
	vectorx<_Tp, _size>& operator+=(const vectorx<_Tp, _size>& rhs);
	//! Item-wise subtraction operator.
	vectorx<_Tp, _size> operator-(const vectorx<_Tp, _size>& rhs) const;
	//! Decrement operator.
	vectorx<_Tp, _size>& operator--();
	//! Item-wise subtraction operator.
	vectorx<_Tp, _size>& operator-=(const vectorx<_Tp, _size>& rhs);
	//! Item-wise multiplication operator.
	vectorx<_Tp, _size> operator*(const vectorx<_Tp, _size>& rhs) const;
	//! Item-wise multiplication operator.
	vectorx<_Tp, _size>& operator*=(const vectorx<_Tp, _size>& rhs);
	//! Division operator.
	vectorx<_Tp, _size> operator/(const vectorx<_Tp, _size>& rhs) const;
	//! Item-wise divide operator.
	vectorx<_Tp, _size>& operator/=(const vectorx<_Tp, _size>& rhs);
	//! Addition operator with scalar.
	vectorx<_Tp, _size> operator+(_Tp rhs) const;
	//! Substraction operator with scalar.
	vectorx<_Tp, _size> operator-(_Tp rhs) const;
	//! Multiplication operator with scalar.
	vectorx<_Tp, _size> operator*(_Tp rhs) const;
	//! Division operator with scalar.
	vectorx<_Tp, _size> operator/(_Tp rhs) const;

	//! Increment operator with scalar.
	vectorx<_Tp, _size>& operator+=(_Tp rhs);
	//! Decrement operator with scalar.
	vectorx<_Tp, _size>& operator-=(_Tp rhs);
	//! Multiplication operator with scalar.
	vectorx<_Tp, _size>& operator*=(_Tp rhs);
	//! Division operator with scalar.
	vectorx<_Tp, _size>& operator/=(_Tp rhs);

	//! Friend operator of additon with scalar.
	friend vectorx<_Tp, _size> operator+(_Tp rhs, const vectorx<_Tp, _size> &lhs) {
		vectorx<_Tp, _size> ret;
		for (unsigned i = 0; i < _size; ++i) {
			ret[i] = rhs + lhs[i];
		}
		return ret;
	}

	//! Friend operator of substraction with scalar.
	friend vectorx<_Tp, _size> operator-(_Tp rhs, const vectorx<_Tp, _size> &lhs) {
		vectorx<_Tp, _size> ret;
		for (unsigned i = 0; i < _size; ++i) {
			ret[i] = rhs - lhs[i];
		}
		return ret;
	}

	//! Friend operator of multiplication with scalar.
	friend vectorx<_Tp, _size> operator*(_Tp rhs, const vectorx<_Tp, _size> &lhs) {
		vectorx<_Tp, _size> ret;
		for (unsigned i = 0; i < _size; ++i) {
			ret[i] = rhs * lhs[i];
		}
		return ret;
	}

	//! Friend operator of division with scalar.
	friend vectorx<_Tp, _size> operator/(_Tp rhs, const vectorx<_Tp, _size> &lhs) {
		vectorx<_Tp, _size> ret;
		for (unsigned i = 0; i < _size; ++i) {
			ret[i] = rhs / lhs[i];
		}
		return ret;
	}
	//! Calculate norm of the vector. By default it's euclidean norm, as in magnitude of a vector.
	real_t norm(Norm n = Norm::L2) const;
	//! Normalize this vector. Default value is euclidean norm.
	void normalize(Norm n = Norm::L2);
	//! Get normalized copy of this vector.
	vectorx normalized(Norm n = Norm::L2) const;
	//! Calculate euclidean distance (L2 norm of vector) to other vector(point).
	real_t distance(const vectorx<_Tp, _size> &lhs, unsigned axis = -1) const;
	//! Calculate sum of each item in vector.
	real_t sum() const;
	//! Calculate mean value of vector.
	real_t mean() const;
	//! Calculate an angle between this and given vector.
	real_t angle(const vectorx<_Tp, _size> &v) const;
	//! Rotate 2-d vector by an angle (in radians).
	vectorx<_Tp, _size>& rotate(real_t angle);
	//! Calculate dot product.
	_Tp dot(const vectorx<_Tp, _size> &rhs) const;
	//! Calculate cross product for 2 and 3 dimensional vector.
	vectorx<_Tp, _size> cross(const vectorx<_Tp, _size> &rhs) const;

	//! Equals operator.
	bool operator==(const vectorx<_Tp, _size>& rhs) const;
	//! Not-equals operator.
	bool operator!=(const vectorx<_Tp, _size>& rhs) const;
	bool operator <(const vectorx<_Tp, _size>& rhs) const;
	//! Index operator.
	_Tp& operator[](unsigned index);
	//! Read-only index operator.
	const _Tp& operator[](unsigned index) const;
	//! Cast to vector of other type.
	template<class _Up> operator vectorx<_Up, _size>() const;
	//! Cast to vector of difference size.
	template<unsigned _new_size> operator vectorx<_Tp, _new_size>() const;

	//! Get bidirectional_iterator at beginning of the vector data.
	iterator begin() {
		return iterator(&_data[0]);
	}
	//! Get bidirectional_iterator at end of the vector data.
	iterator end() {
		return iterator(&_data[_size]);
	}
	//! Get read-only bidirectional_iterator at beginning of the vector data.
	const_iterator begin() const {
		return const_iterator(&_data[0]);
	}
	//! Get read-only bidirectional_iterator at end of the vector data.
	const_iterator end() const {
		return const_iterator(&_data[_size]);
	}
	//! Get vector with values ranged from inside this vector. Does not reference data, but makes a copy.
	template<unsigned range_data, unsigned range_end>
	vectorx<_Tp, range_end - range_data> range();

	//! ostream operator.
	friend std::ostream& operator<<(std::ostream& stream, const vectorx<_Tp, _size> &arr) {
		stream << arr[0];
		for (unsigned i = 1; i < _size; i++) {
			stream <<" " << arr[i];
		}
		return stream;
	}
};

typedef vectorx<float, 2> vec2f;
typedef vectorx<float, 3> vec3f;
typedef vectorx<float, 4> vec4f;
typedef vectorx<float, 6> vec6f;

typedef vectorx<double, 2> vec2d;
typedef vectorx<double, 3> vec3d;
typedef vectorx<double, 4> vec4d;
typedef vectorx<double, 6> vec6d;

typedef vectorx<int, 2> vec2i;
typedef vectorx<int, 3> vec3i;
typedef vectorx<int, 4> vec4i;
typedef vectorx<int, 6> vec6i;

typedef vectorx<unsigned, 2> vec2ui;
typedef vectorx<unsigned, 3> vec3ui;
typedef vectorx<unsigned, 4> vec4ui;
typedef vectorx<unsigned, 6> vec6ui;

typedef vectorx<short, 2> vec2s;
typedef vectorx<short, 3> vec3s;
typedef vectorx<short, 4> vec4s;
typedef vectorx<short, 6> vec6s;

typedef vectorx<unsigned char, 2> vec2b;
typedef vectorx<unsigned char, 3> vec3b;
typedef vectorx<unsigned char, 4> vec4b;
typedef vectorx<unsigned char, 6> vec6b;

typedef vec2i point2i;
typedef vec2d point2d;
typedef vec2f point2f;
typedef vec2b point2b;

typedef vec3i point3i;
typedef vec3d point3d;
typedef vec3f point3f;
typedef vec3b point3b;

typedef vec2ui size2;
typedef vec3ui size3;

typedef point2i point;

typedef vec3b color3;
typedef vec4b color4;

typedef vector<int> vectori; //!< 32-bit int vector
typedef vector<unsigned> vectorui; //!< 32-bit unsigned int vector
typedef vector<float> vectorf; //!< 32-bit float vector.
typedef vector<double> vectord; //!< 64-bit float vector.
typedef vector<unsigned char> vectorb; //!< 8-bit unsigned char vector (byte).
typedef vector<short> vectors; //!< 16-bit short vector.

typedef vector<vec2d> vector2d; //!< Two channel 64-bit double vector
typedef vector<vec2i> vector2i; //!< Two channel 32-bit int vector
typedef vector<vec2f> vector2f; //!< Two channel 32-bit float vector
typedef vector<vec2s> vector2s; //!< Two channel 16-bit short vector
typedef vector<vec2b> vector2b; //!< Two channel 8-bit byte vector

typedef vector<vec3d> vector3d; //!< Three channel 64-bit double vector
typedef vector<vec3i> vector3i; //!< Three channel 32-bit int vector
typedef vector<vec3f> vector3f; //!< Three channel 32-bit float vector
typedef vector<vec3s> vector3s; //!< Three channel 16-bit short vector
typedef vector<vec3b> vector3b; //!< Three channel 8-bit byte vector

typedef vector<vec4d> vector4d; //!< Four channel 64-bit double vector
typedef vector<vec4i> vector4i; //!< Four channel 32-bit int vector
typedef vector<vec4f> vector4f; //!< Four channel 32-bit float vector
typedef vector<vec4s> vector4s; //!< Four channel 16-bit short vector
typedef vector<vec4b> vector4b; //!< Four channel 8-bit byte vector

#ifdef CV_REAL_TYPE_DOUBLE
typedef vec2d vec2r;
typedef vec3d vec3r;
typedef vec4d vec4r;
typedef vec6d vec6r;

typedef vectord vectorr;
typedef vector2d vector2r;
typedef vector3d vector3r;
typedef vector4d vector4r;
#else
typedef vec2f vec2r;
typedef vec3f vec3r;
typedef vec4f vec4r;
typedef vec6f vec6r;

typedef vectorf vectorr;
typedef vector2f vector2r;
typedef vector3f vector3r;
typedef vector4f vector4r;
#endif

// Implementation

template<class _Tp>
vector<_Tp>::vector():
	super_type() {
}

template<class _Tp>
vector<_Tp>::vector(pointer data, pointer begin, unsigned length, unsigned stride, refcount_type *refcount) {
	ASSERT(length > 0);
	this->_refcount = refcount;
	this->_data = data;
	this->_begin = begin;
	if (refcount) REF_INCREMENT(refcount);
	this->_shape = {length};
	this->_strides = {stride};
}

template<class _Tp>
vector<_Tp>::vector(pointer data, unsigned length, unsigned stride, bool borrow) : super_type() {
	ASSERT(length && stride <= length);
	length = static_cast<unsigned>(std::ceil(static_cast<double>(length) / stride));
	if (borrow) {
		this->_data = data;
		this->_begin = data;
		this->_shape = {length};
		this->_strides = {stride};
	} else {
		this->allocate({length});
		for (int i = 0; i < length; ++i) {
			this->_begin[i] = data[i*stride];
		}
		this->_refcount = REF_NEW;
	}
}

template<class _Tp>
vector<_Tp>::vector(unsigned size):
	super_type(size) {
}

template<class _Tp>
vector<_Tp>::vector(const std::initializer_list<_Tp> &list) :
	basic_array<_Tp>(list.size()) {
	std::copy(list.begin(), list.end(), this->_begin);
}

template<class _Tp>
vector<_Tp>::vector(const vector &cpy, bool deepCopy):
	super_type() {
	this->copy(cpy, deepCopy);
}

template<class _Tp>
vector<_Tp>::vector(vector &&move):
	super_type(move) {
}

template<class _Tp>
vector<_Tp>::~vector() {
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator=(vector &&rhs) {
	if (this != &rhs) {
		basic_array<_Tp>::operator=(rhs);
	}
	return *this;
}
template<class _Tp>
void vector<_Tp>::create(unsigned size) {
	if (this->length() == size) {
		return;
	}
	this->release();
	this->allocate({size});
}

template<class _Tp>
void vector<_Tp>::create(const vector &from, bool deepCopy) {
	this->copy(from, deepCopy);
}

template<class _Tp>
vector<_Tp> vector<_Tp>::clone() const {
	return vector<_Tp>(*this, true);
}

template<class _Tp>
template<class _Up>
vector<_Tp>::operator vector<_Up>() const {
	static_assert(std::is_convertible< _Tp, _Up >::value, "vector::convert_to!~ Types are not convertible.");
	if (!(*this))
		return vector<_Up>();

	vector<_Up> ret (this->length());

	for (unsigned i = 0; i < this->length(); ++i) {
		ret[i] = static_cast<_Up>((*this)[i]);
	}

	return ret;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator()(unsigned start, unsigned end, unsigned stride) {
	ASSERT(start <= end && start < this->length() && start >= 0 && end < this->length() && end >= 0);

	vector<_Tp> vec;

	vec._data = this->_data;
	vec._begin = this->_begin + start;
	vec._shape = {static_cast<unsigned>(std::ceil(static_cast<double>(end - start + 1) / stride))};
	vec._strides = this->_strides;
	vec._strides[0]*=stride;
	vec._refcount = this->_refcount;

	REF_INCREMENT(this->_refcount);

	return vec;
}

template<class _Tp>
const vector<_Tp> vector<_Tp>::operator()(unsigned start, unsigned end, unsigned stride) const {
	ASSERT(start <= end && start < this->length() && start >= 0 && end < this->length() && end >= 0);

	vector<_Tp> vec;

	vec._data = this->_data;
	vec._begin = this->_begin + start;
	vec._shape = {static_cast<unsigned>(std::ceil(static_cast<double>(end - start + 1) / stride))};
	vec._strides = this->_strides;
	vec._strides[0]*=stride;
	vec._refcount = this->_refcount;

	REF_INCREMENT(this->_refcount);

	return vec;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::zeros(unsigned count) {
	vector<_Tp> vec(count);
	vec.construct(0);
	return vec;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::ones(unsigned count) {
	vector<_Tp> vec(count);
	vec.construct(1);
	return vec;
}

template<class _Tp>
CV_TYPENAME vector<_Tp>::iterator vector<_Tp>::min() {
	return std::min_element(this->begin(), this->end());
}

template<class _Tp>
CV_TYPENAME vector<_Tp>::iterator vector<_Tp>::max() {
	return std::max_element(this->begin(), this->end());
}

template<class _Tp>
CV_TYPENAME vector<_Tp>::const_iterator vector<_Tp>::min() const {
	return std::min_element(this->begin(), this->end());
}

template<class _Tp>
CV_TYPENAME vector<_Tp>::const_iterator vector<_Tp>::max() const {
	return std::max_element(this->begin(), this->end());
}

template<class _Tp>
void vector<_Tp>::fill(const _Tp &value) {
	if (this->is_contiguous())
		std::memset(this->_begin, value, this->length()*sizeof(_Tp));
	else {
		for (auto &v : *this) {
			v = value;
		}
	}
}

template<class _Tp>
_Tp vector<_Tp>::dot(const vector<_Tp> &rhs) const {
	ASSERT(this->length() == rhs.length());
	_Tp d = 0;
	for (unsigned i = 0; i < this->length(); ++i) {
		d += this->_data[i] * rhs[i];
	}
	return d;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::cross(const vector<_Tp> &rhs) const {
	ASSERT(this->length() == rhs.length() && (this->length() == 2 || this->length() == 3));
	vector<_Tp> c(this->length());
	if (this->length() == 2) {
		c[0] = (*this)[0] - rhs[1];
		c[1] =  rhs[0] - (*this)[1];
	} else {
		c[0] = (*this)[1]*rhs[2] - (*this)[2]*rhs[1];
		c[1] = -1*((*this)[0]*rhs[2] - (*this)[2]*rhs[0]);
		c[2] = (*this)[0]*rhs[1] - (*this)[1]*rhs[0];
	}
	return c;
}

template<class _Tp>
void vector<_Tp>::sort() {
	std::sort(this->begin(), this->end());
}

template<class _Tp>
real_t vector<_Tp>::norm(Norm n) const {
	return internal::norm(this->begin(), this->end(), n);
}

template<class _Tp>
void vector<_Tp>::normalize(Norm n) {
	internal::normalize(this->begin(), this->end(), n);
}

template<class _Tp>
vector<_Tp> vector<_Tp>::normalized(Norm n) const {
	auto v = this->clone();
	v.normalize(n);
	return v;
}

template<class _Tp>
real_t vector<_Tp>::distance(const vector<_Tp> &rhs, Norm n) const {
	ASSERT(this->length() == rhs.length());
	real_t d = 0;

	switch (n) {
	case Norm::L1: {
		for (unsigned i = 0; i < this->length(); ++i) {
			d += std::fabs(static_cast<double>(this->at_index(i) - rhs[i]));
		}
	}
	break;
	case Norm::L2: {
		for (unsigned i = 0; i < this->length(); ++i) {
			d += pow(this->at_index(i) - rhs[i], 2);
		}
		d = std::sqrt(d);
	}
	break;
	default:
		throw std::runtime_error("Norm type not supported");
	}
	return d;
}

template<class _Tp>
template<class _Cmp>
void vector<_Tp>::sort(_Cmp cmp) {
	std::sort(this->begin(), this->end(), cmp);
}

template<class _Tp>
CV_TYPENAME vector<_Tp>::reference vector<_Tp>::operator [](int index) {
	index = INTERPRET_INDEX(index, this->length());
	ASSERT(index < this->length());
	return this->at_index(index);
}

template<class _Tp>
CV_TYPENAME vector<_Tp>::const_reference vector<_Tp>::operator [](int index) const {
	index = INTERPRET_INDEX(index, this->length());
	ASSERT(index < this->length());
	return this->at_index(index);
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator+(const vector<_Tp> &rhs) const {
	vector<_Tp> retVal(*this, true);
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.size());
		for (int i = 0; i < min_size; ++i) {
			retVal[i] += rhs[i];
		}
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator-(const vector<_Tp> &rhs) const {
	vector<_Tp> retVal(*this, true);
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			retVal[i] -= rhs[i];
		}
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator*(const vector<_Tp> &rhs) const {
	vector<_Tp> retVal(*this, true);
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			retVal[i] *= rhs[i];
		}
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator/(const vector<_Tp> &rhs) const {
	vector<_Tp> retVal(*this, true);
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			retVal[i] /= rhs[i];
		}
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator+=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			(*this)[i] += rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator-=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			(*this)[i] -= rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator*=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			(*this)[i] *= rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator/=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		for (int i = 0; i < min_size; ++i) {
			(*this)[i] /= rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator+(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	for(unsigned i = 0; i < this->length(); ++i) {
		retVal[i] += rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator-(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	for(unsigned i = 0; i < this->length(); ++i) {
		retVal[i] -= rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator*(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	for(unsigned i = 0; i < this->length(); ++i) {
		retVal[i] *= rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator/(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	for(unsigned i = 0; i < this->length(); ++i) {
		retVal[i] /= rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator+=(const _Tp &rhs) {
	for(unsigned i = 0; i < this->length(); ++i) {
		(*this)[i] += rhs;
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator-=(const _Tp &rhs) {
	for(unsigned i = 0; i < this->length(); ++i) {
		(*this)[i] -= rhs;
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator*=(const _Tp &rhs) {
	for(unsigned i = 0; i < this->length(); ++i) {
		(*this)[i] *= rhs;
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator/=(const _Tp &rhs) {
	for(unsigned i = 0; i < this->length(); ++i) {
		(*this)[i] /= rhs;
	}
	return *this;
}

template<class _Tp>
bool vector<_Tp>::operator==(const vector<_Tp> &rhs) const {
	if (this->length() != rhs.length()) {
		return false;
	} else {
	for(unsigned i = 0; i < this->length(); ++i) {
			if ((*this)[i] != rhs[i]) {
				return false;
			}
		}
		return true;
	}
}

template<class _Tp>
bool vector<_Tp>::operator !=(const vector<_Tp> &rhs) const {
	return !operator==(rhs);
}

template<class _Tp>
bool vector<_Tp>::operator <(const vector<_Tp> &rhs) const {
	ASSERT(!this->empty() && !rhs.empty());
	return std::lexicographical_compare(this->begin(), this->end(), rhs.begin(), rhs.end());
}

template<class _Tp>
vector<_Tp>::operator bool() const {
	return (this->_data);
}

///////////////////////////////////////////////////////////////////////////////
/// class: vectorx

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx() {
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const std::initializer_list<_Tp> &list) {
	ASSERT(list.size() == _size);
	for (unsigned i = 0; i < _size; ++i) {
		this->_data[i] = *(list.begin() + i);
	}
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp &a) {
	for (int i = 0; i < _size; ++i) {
		_data[i] = a;
	}
}

//! Vector-2 initializer. Vector needs to be previously defined as length 2.
template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp & a, const _Tp & b) {
	static_assert(_size == 2, " Vector needs to be previously defined as length 2");
	_data[0] = a;
	_data[1] = b;
}

//! Vector-3 initializer. Vector needs to be previously defined as length 3.
template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp & a, const _Tp & b, const _Tp & c) {
	static_assert(_size == 3, " Vector needs to be previously defined as length 3");
	_data[0] = a;
	_data[1] = b;
	_data[2] = c;
}

//! Vector-4 initializer. Vector needs to be previously defined as length 4.
template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp & a, const _Tp & b, const _Tp & c, const _Tp & d) {
	static_assert(_size == 4, " Vector needs to be previously defined as length 4");
	_data[0] = a;
	_data[1] = b;
	_data[2] = c;
	_data[3] = d;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const vectorx<_Tp, _size> &cpy) {
	for (unsigned i = 0; i < _size; ++i) {
		this->_data[i] = cpy[i];
	}
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const vectorx<_Tp, _size / 2> &rhs, const vectorx<_Tp, _size / 2> &lhs) {
	for(unsigned i = 0; i < _size / 2; ++i) {
		this->_data[i] = rhs[i];
	}
	for(unsigned i = _size / 2; i< _size; ++i) {
		this->_data[i] = lhs[i - _size / 2];
	}
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::~vectorx() {
}

template<class _Tp, unsigned _size>
_Tp *vectorx<_Tp, _size>::data() {
	return this->_data;
}

template<class _Tp, unsigned _size>
unsigned vectorx<_Tp, _size>::size() const {
	return _size;
}

template<class _Tp, unsigned _size>
unsigned vectorx<_Tp, _size>::byte_size() const {
	return _size * sizeof(_Tp);
}

template<class _Tp, unsigned _size>
template<class _Up>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator=(const _Up & rhs) {
	for (int i = 0; i < _size; ++i) {
		this->_data[i] = static_cast<_Up>(rhs);
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator=(const vectorx<_Tp, _size> &rhs) {
	if (this != &rhs) {
		for (int i = 0; i < _size; ++i) {
			this->_data[i] = rhs[i];
		}
	}
	return *this;
}

template<class _Tp, unsigned _size>
template<unsigned o_size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator=(const vectorx<_Tp, o_size> &rhs) {
	unsigned lesser = std::min(_size, o_size);
	if (this != &rhs) {
		for (unsigned i = 0; i < lesser; ++i) {
			this->_data[i] = rhs[i];
		}
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator+(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	for (int i = 0; i < _size; ++i) {
		ret[i] = rhs._data[i] + this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator++() {
	for (int i = 0; i < _size; ++i) {
		this->_data[i]++;
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator+=(const vectorx<_Tp, _size>& rhs) {
	for (int i = 0; i < _size; ++i) {
		this->_data[i] += rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator-(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	for (int i = 0; i < _size; ++i) {
		ret[i] = rhs._data[i] - this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator--() {
	for (int i = 0; i < _size; ++i) {
		this->_data[i]--;
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator-=(const vectorx<_Tp, _size>& rhs) {
	for (int i = 0; i < _size; ++i) {
		this->_data[i] -= rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator*(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	for (int i = 0; i < _size; ++i) {
		ret[i] = rhs._data[i] * this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator*=(const vectorx<_Tp, _size>& rhs) {
	for (int i = 0; i < _size; ++i) {
		this->_data[i] *= rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator/(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	for (int i = 0; i < _size; ++i) {
		ret[i] = rhs._data[i] / this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator/=(const vectorx<_Tp, _size>& rhs) {
	for (int i = 0; i < _size; ++i) {
		this->_data[i] /= rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator+(_Tp rhs) const {
	vectorx<_Tp, _size> ret;
	for (unsigned i = 0; i < _size; ++i) {
		ret[i] = this->_data[i] + rhs;
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator-(_Tp rhs) const {
	vectorx<_Tp, _size> ret;
	for (unsigned i = 0; i < _size; ++i) {
		ret[i] = this->_data[i] - rhs;
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator*(_Tp rhs) const {
	vectorx<_Tp, _size> ret;
	for (unsigned i = 0; i < _size; ++i) {
		ret[i] = this->_data[i] * rhs;
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator/(_Tp rhs) const {
	vectorx<_Tp, _size> ret;
	for (unsigned i = 0; i < _size; ++i) {
		ret[i] = this->_data[i] / rhs;
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator+=(_Tp rhs) {
	for (unsigned i = 0; i < _size; ++i) {
		this->_data[i] += rhs;
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator-=(_Tp rhs) {
	for (unsigned i = 0; i < _size; ++i) {
		this->_data[i] -= rhs;
	}
	return *this;
}


template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator*=(_Tp rhs) {
	for (unsigned i = 0; i < _size; ++i) {
		this->_data[i] *= rhs;
	}
	return *this;
}


template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator/=(_Tp rhs) {
	for (unsigned i = 0; i < _size; ++i) {
		this->_data[i] /= rhs;
	}
	return *this;
}



template<class _Tp, unsigned _size>
real_t vectorx<_Tp, _size>::norm(Norm n) const {
	return internal::norm(this->_data, this->_data + _size, n);
}

template<class _Tp, unsigned _size>
void vectorx<_Tp, _size>::normalize(Norm n) {
	internal::normalize(this->_data, this->_data + _size, n);
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::normalized(Norm n) const {
	auto r = *this;
	r.normalize(n);
	return r;
}

template<class _Tp, unsigned _size>
real_t vectorx<_Tp, _size>::distance(const vectorx<_Tp, _size> &lhs, unsigned axis) const {
	real_t distVal = 0;
	if (axis == -1) {
		for (int i = 0; i < _size; ++i) {
			distVal += pow((*this)[i] - lhs[i], 2);
		}
		return std::sqrt(distVal);
	} else {
		ASSERT(axis < _size);
		return sqrt(pow(this->_data[axis], 2) - pow(lhs[axis], 2));
	}
}

template<class _Tp, unsigned _size>
real_t vectorx<_Tp, _size>::sum() const {
	real_t retVal = 0;
	for (int i = 0; i < _size; ++i) {
		retVal += (real_t)this->_data[i];
	}
	return retVal;
}

template<class _Tp, unsigned _size>
real_t vectorx<_Tp, _size>::mean() const {
	return (sum() / _size);
}

template<class _Tp, unsigned _size>
real_t vectorx<_Tp, _size>::angle(const vectorx<_Tp, _size> &v) const {

	real_t ang;

	if ((*this).norm() == 0.0 || v.norm() == 0.0) {
		ang = 0;
	} else {
		ang = acos((this->dot(v)) / (this->norm() * v.norm()));
	}

	return ang;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::rotate(real_t angle) {
	static_assert(_size == 2, "Rotation is only allowed for 2D vectors.");
	_Tp new_x, new_y;

	new_x = this->_data[0] * cos(angle) - this->_data[1] * sin(angle);
	new_y = this->_data[0] * sin(angle) + this->_data[1] * cos(angle);

	this->x() = new_x;
	this->y() = new_y;

	return *this;
}

template<class _Tp, unsigned _size>
_Tp vectorx<_Tp, _size>::dot(const vectorx<_Tp, _size> &rhs) const {
	_Tp d = 0;
	for (unsigned i = 0; i < _size; ++i) {
		d += this->_data[i] * rhs[i];
	}
	return d;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::cross(const vectorx<_Tp, _size> &rhs) const {
	static_assert(_size == 2 || _size == 3, "Cross product only applies to 2D and 3D vectors");
	vectorx<_Tp, _size> c;
	if (_size == 2) {
		c[0] = this->_data[0] - rhs._data[1];
		c[1] =  rhs._data[0] - this->_data[1];
	} else {
		c[0] = this->_data[1]*rhs._data[2] - this->_data[2]*rhs._data[1];
		c[1] = -1*(this->_data[0]*rhs._data[2] - this->_data[2]*rhs._data[0]);
		c[2] = this->_data[0]*rhs._data[1] - this->_data[1]*rhs._data[0];
	}
	return c;
}

template<class _Tp, unsigned _size>
bool vectorx<_Tp, _size>::operator==(const vectorx<_Tp, _size>& rhs) const {
	for (int i = 0; i < _size; ++i) {
		if (this->_data[i] != rhs._data[i])
			return false;
	}
	return true;
}

template<class _Tp, unsigned _size>
bool vectorx<_Tp, _size>::operator!=(const vectorx<_Tp, _size>& rhs) const {
	return !operator==(rhs);
}

template<class _Tp, unsigned _size>
bool vectorx<_Tp, _size>::operator <(const vectorx<_Tp, _size>& rhs) const {
	return std::lexicographical_compare(this->begin(), this->end(), rhs.begin(), rhs.end());
}

template<class _Tp, unsigned _size>
_Tp& vectorx<_Tp, _size>::operator[](unsigned index) {
	ASSERT(index < _size);
	return this->_data[index];
}

template<class _Tp, unsigned _size>
const _Tp& vectorx<_Tp, _size>::operator[](unsigned index) const {
	ASSERT(index < _size);
	return this->_data[index];
}

template<class _Tp, unsigned _size>
template<class _Up>
vectorx<_Tp, _size>::operator vectorx<_Up, _size>() const {
	vectorx<_Up, _size> new_array;
	for (int i = 0; i < _size; ++i) {
		new_array[i] = static_cast<_Up>(this->_data[i]);
	}
	return new_array;
}

template<class _Tp, unsigned _size>
template<unsigned _new_size>
vectorx<_Tp, _size>::operator vectorx<_Tp, _new_size>() const {
	vectorx<_Tp, _new_size> newVec;
	unsigned lesser = std::min(_size, _new_size);
	for (unsigned i = 0; i < lesser; ++i) {
		newVec[i] = this->_data[i];
	}
	return newVec;
}

template<class _Tp, unsigned _size>
template<unsigned range_start, unsigned range_end>
vectorx<_Tp, range_end - range_start> vectorx<_Tp, _size>::range() {
	static_assert(range_end < _size, "Invalid range end.");
	vectorx<_Tp, range_end - range_start> retVal;
	for (int i = range_start, retIndex = 0; i < range_end; i++, retIndex++) {
		retVal[retIndex] = this->_data[i];
	}
	return retVal;
}

}
#endif /* end of include guard: VECTOR_HPP_TRD8NTPW */


