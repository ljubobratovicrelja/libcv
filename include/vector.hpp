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


#ifndef VECTOR_HPP_TRD8NTPW
#define VECTOR_HPP_TRD8NTPW


#include "array.hpp"


namespace cv {

/*!
 * @brief vector class with data allocated on heap.
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

	//! Calculate norm of the vector.
	double norm(Norm n = Norm::L2) const;
	//! Calculate distance of two vectors. Need to be of same size.
	double distance(const vector &rhs, Norm n = Norm::L2) const;

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
		for (int i = 0; i < arr.length(); i++) {
			stream << arr[i] << " ";
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
public:
	typedef random_access_iterator<_Tp> iterator;  //!< random access iterator.
	typedef random_access_iterator<const _Tp> const_iterator; //!< read-only random access iterator.

protected:
	_Tp _data[_size];  //!< Data of the vector.

public:
	//! Default constructor.
	vectorx();
	vectorx(const std::initializer_list<_Tp> &list);

	//! Constructor using scalar value - fills vector with it.
	vectorx(const _Tp &a);
	//! Construct a 2D vector.
	vectorx(const _Tp &a, const _Tp &b);
	//! Construct a 3D vector.
	vectorx(const _Tp &a, const _Tp &b, const _Tp &c);
	//! Construct a 4D vector.
	vectorx(const _Tp &a, const _Tp &b, const _Tp &c, const _Tp &d);
	//! Copy constructor.
	vectorx(const vectorx<_Tp, _size> &cpy);

	//! Merging constructor - merges two half-sized vectors to one.
	vectorx(const vectorx<_Tp, _size / 2> &rhs, const vectorx<_Tp, _size / 2> &lhs);

	//! Class destructor.
	~vectorx();

	//! Get pointer to the beginning of the vector data.
	_Tp *getData();

	//! Get item size(count) of the vector.
	unsigned size() const;

	//! Get byte size of the vector. It is defined as item count * sizeof(value_type)
	unsigned byteSize() const;

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

	//! Calculate eucledian distance (L2 norm of vector) to other vector(point).
	double distance(const vectorx<_Tp, _size> &lhs, unsigned axis = -1) const;
	//! Calculate sum of each item in vector.
	double sum() const;
	//! Calculate mean value of vector.
	double mean() const;
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
		for (int i = 0; i < _size; i++) {
			stream << arr[i] << " ";
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
	ASSERT(this->length());

	start = INTERPRET_INDEX(start, this->length());
	end = INTERPRET_INDEX(end, this->length());

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
	ASSERT(this->length());

	start = INTERPRET_INDEX(start, this->length());
	end = INTERPRET_INDEX(end, this->length());

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
void vector<_Tp>::sort() {
	std::sort(this->begin(), this->end());
}

template<class _Tp>
double vector<_Tp>::norm(Norm n) const {
	double nv = 0;
	switch (n) {
		case Norm::L1:
			for (auto v : *this) {
				nv += std::fabs(v);
			}
			break;
		case Norm::L2:
			for (auto v : *this) {
				nv += v*v;
			}
			nv = sqrt(nv);
			break;
		default:
			throw std::runtime_error("Norm type not supported");
	}
	return nv;
}

template<class _Tp>
void vector<_Tp>::normalize(Norm n) {
	auto nv = this->norm(n);
	for (auto &v : *this) {
		v /= nv;
	}
}

template<class _Tp>
vector<_Tp> vector<_Tp>::normalized(Norm n) const {
	auto v = this->clone();
	v.normalize(n);
	return v;
}

template<class _Tp>
double vector<_Tp>::distance(const vector<_Tp> &rhs, Norm n) const {
	ASSERT(this->length() == rhs.length());
	double d = 0;

	switch (n) {
		case Norm::L1:
			{
				for (unsigned i = 0; i < this->length(); ++i) {
					d += std::fabs(this->at_index(i) - rhs[i]);
				}
			}
			break;
		case Norm::L2:
			{
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
		LOOP_FOR_TO(min_size) {
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
		LOOP_FOR_TO(min_size) {
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
		LOOP_FOR_TO(min_size) {
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
		LOOP_FOR_TO(min_size) {
			retVal[i] /= rhs[i];
		}
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator+=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		LOOP_FOR_TO(min_size) {
			(*this)[i] += rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator-=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		LOOP_FOR_TO(min_size) {
			(*this)[i] -= rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator*=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		LOOP_FOR_TO(min_size) {
			(*this)[i] *= rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator/=(const vector<_Tp> &rhs) {
	if (this->_data && rhs._data) {
		int min_size = std::min(this->length(), rhs.length());
		LOOP_FOR_TO(min_size) {
			(*this)[i] /= rhs[i];
		}
	}
	return *this;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator+(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	LOOP_FOR_TO(this->length()) {
		retVal[i] += rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator-(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	LOOP_FOR_TO(this->length()) {
		retVal[i] -= rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator*(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	LOOP_FOR_TO(this->length()) {
		retVal[i] *= rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> vector<_Tp>::operator/(const _Tp &rhs) const {
	vector<_Tp> retVal(*this, true);
	LOOP_FOR_TO(this->length()) {
		retVal[i] /= rhs;
	}
	return retVal;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator+=(const _Tp &rhs) {
	LOOP_FOR_TO(this->length()) {
		(*this)[i] += rhs;
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator-=(const _Tp &rhs) {
	LOOP_FOR_TO(this->length()) {
		(*this)[i] -= rhs;
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator*=(const _Tp &rhs) {
	LOOP_FOR_TO(this->length()) {
		(*this)[i] *= rhs;
	}
	return *this;
}

template<class _Tp>
vector<_Tp> &vector<_Tp>::operator/=(const _Tp &rhs) {
	LOOP_FOR_TO(this->length()) {
		(*this)[i] /= rhs;
	}
	return *this;
}

template<class _Tp>
bool vector<_Tp>::operator==(const vector<_Tp> &rhs) const {
	if (this->length() != rhs.length()) {
		return false;
	} else {
		LOOP_FOR_TO(this->length()) {
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
	LOOP_FOR_TO(_size)
		this->_data[i] = *(list.begin() + i);
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp &a) {
	LOOP_FOR_TO(_size)
		this->_data[i] = a;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp &a, const _Tp &b) {
	static_assert(_size == 2, "Invalid vector size-constructor.");
	this->_data[0] = a;
	this->_data[1] = b;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp &a, const _Tp &b, const _Tp &c) {
	static_assert(_size == 3, "Invalid vector size-constructor.");
	this->_data[0] = a;
	this->_data[1] = b;
	this->_data[2] = c;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const _Tp &a, const _Tp &b, const _Tp &c, const _Tp &d) {
	static_assert(_size == 4, "Invalid vector size-constructor.");
	this->_data[0] = a;
	this->_data[1] = b;
	this->_data[2] = c;
	this->_data[3] = d;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const vectorx<_Tp, _size> &cpy) {
	LOOP_FOR(0, _size, 1) {
		this->_data[i] = cpy[i];
	}
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::vectorx(const vectorx<_Tp, _size / 2> &rhs, const vectorx<_Tp, _size / 2> &lhs) {
	LOOP_FOR(0, _size / 2, 1) {
		this->_data[i] = rhs[i];
	}
	LOOP_FOR(_size / 2, _size, 1) {
		this->_data[i] = rhs[i - _size / 2];
	}
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>::~vectorx() {
}

template<class _Tp, unsigned _size>
_Tp *vectorx<_Tp, _size>::getData() {
	return this->_data;
}

template<class _Tp, unsigned _size>
unsigned vectorx<_Tp, _size>::size() const {
	return _size;
}

template<class _Tp, unsigned _size>
unsigned vectorx<_Tp, _size>::byteSize() const {
	return _size * sizeof(_Tp);
}

template<class _Tp, unsigned _size>
template<class _Up>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator=(const _Up & rhs) {
	LOOP_FOR(0, _size, 1) {
		this->_data[i] = static_cast<_Up>(rhs);
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator=(const vectorx<_Tp, _size> &rhs) {
	if (this != &rhs) {
		LOOP_FOR(0, _size, 1) {
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
		LOOP_FOR_TO(lesser) {
			this->_data[i] = rhs[i];
		}
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator+(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	LOOP_FOR(0, _size, 1) {
		ret = rhs._data[i] + this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator++() {
	LOOP_FOR(0, _size, 1) {
		this->_data[i]++;
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator+=(const vectorx<_Tp, _size>& rhs) {
	LOOP_FOR(0, _size, 1) {
		this->_data[i] += rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator-(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	LOOP_FOR(0, _size, 1) {
		ret = rhs._data[i] - this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator--() {
	LOOP_FOR(0, _size, 1) {
		this->_data[i]--;
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator-=(const vectorx<_Tp, _size>& rhs) {
	LOOP_FOR(0, _size, 1) {
		this->_data[i] -= rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator*(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	LOOP_FOR(0, _size, 1) {
		ret = rhs * this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator*=(const vectorx<_Tp, _size>& rhs) {
	LOOP_FOR(0, _size, 1) {
		this->_data[i] *= rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size> vectorx<_Tp, _size>::operator/(const vectorx<_Tp, _size>& rhs) const {
	vectorx<_Tp, _size> ret;
	LOOP_FOR(0, _size, 1) {
		ret = rhs._data[i] / this->_data[i];
	}
	return ret;
}

template<class _Tp, unsigned _size>
vectorx<_Tp, _size>& vectorx<_Tp, _size>::operator/=(const vectorx<_Tp, _size>& rhs) {
	LOOP_FOR(0, _size, 1) {
		this->_data[i] /= rhs._data[i];
	}
	return *this;
}

template<class _Tp, unsigned _size>
double vectorx<_Tp, _size>::distance(const vectorx<_Tp, _size> &lhs, unsigned axis) const {
	double distVal = 0;
	if (axis == -1) {
		LOOP_FOR_TO(_size) {
			distVal += pow((*this)[i] - lhs[i], 2);
		}
		return std::sqrt(distVal);
	} else {
		ASSERT(axis < _size);
		return sqrt(pow(this->_data[axis], 2) - pow(lhs[axis], 2));
	}
}

template<class _Tp, unsigned _size>
double vectorx<_Tp, _size>::sum() const {
	double retVal = 0;
	LOOP_FOR(0, _size, 1) {
		retVal += (double)this->_data[i];
	}
	return retVal;
}

template<class _Tp, unsigned _size>
double vectorx<_Tp, _size>::mean() const {
	return (sum() / _size);
}

template<class _Tp, unsigned _size>
bool vectorx<_Tp, _size>::operator==(const vectorx<_Tp, _size>& rhs) const {
	LOOP_FOR(0, _size, 1) {
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
	return this->_data[index];
}

template<class _Tp, unsigned _size>
const _Tp& vectorx<_Tp, _size>::operator[](unsigned index) const {
	return this->_data[index];
}

template<class _Tp, unsigned _size>
template<class _Up>
vectorx<_Tp, _size>::operator vectorx<_Up, _size>() const {
	vectorx<_Up, _size> new_array;
	LOOP_FOR(0, _size, 1) {
		new_array[i] = static_cast<_Up>(this->_data[i]);
	}
	return new_array;
}

template<class _Tp, unsigned _size>
template<unsigned _new_size>
vectorx<_Tp, _size>::operator vectorx<_Tp, _new_size>() const {
	vectorx<_Tp, _new_size> newVec;
	unsigned lesser = std::min(_size, _new_size);
	LOOP_FOR_TO(lesser) {
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
