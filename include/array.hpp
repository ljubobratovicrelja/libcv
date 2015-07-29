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
// Contains basic n-d array implementation, used as base class to any other
// array-like structure, such as vector, matrix, etc.
//
// Authod:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef ARRAY_HPP_MVWJDTQ1
#define ARRAY_HPP_MVWJDTQ1


#include <iostream>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstring>

#include "fwd.hpp"
#include "indexarray.hpp"
#include "iterator.hpp"
#include "parallelization.hpp"


namespace cv {

/*!
 * @brief Base array class with reference counted memory.
 *
 */
template<class _Tp>
class basic_array {
public:
	typedef _Tp value_type;
	typedef _Tp &reference;
	typedef const _Tp &const_reference;
	typedef _Tp *pointer;
	typedef const _Tp *const_pointer;
	typedef unsigned size_type;
	typedef std::ptrdiff_t difference_type;

protected:
	pointer _data; //!< Memory of the array.
	pointer _begin; //!< Beginning of the array data.
	index_array _shape; //!< Shape of the array.
	index_array _strides; //!< Strides of the array.
	refcount_type *_refcount; //!< Reference counter.

	//! Allocate array of given size.
	virtual void allocate(const index_array &shape, unsigned byte_size = 1);
	//! Deallocate array memory.
	virtual void deallocate();
	//! Copy the array from another.
	void copy(const basic_array &cpy, bool deepCopy);
	//! Perform the move.
	void move(basic_array &&move);

	//! Default (null) constructor.
	basic_array();
	//! Class constructor from existing reference counted data.
	basic_array(pointer data, pointer begin, const index_array &shape, const index_array &strides, refcount_type *refcount);
	//! 1D array constructor.
	basic_array(size_type size);
	//! 2D array constructor.
	basic_array(size_type size1, size_type size2);
	//! 3D array constructor.
	basic_array(size_type size1, size_type size2, size_type size3);

	//! Copy constructor.
	basic_array(const basic_array &cpy, bool deepCopy = false);
	//! Move constructor.
	basic_array(basic_array &&move);

	//! Index 1-d array at index i.
	reference at_index(unsigned i);
	//! Read-only index 1-d array at index i.
	const_reference at_index(unsigned i) const;

	//! Index 2-d array at index i, j.
	reference at_index(unsigned i, unsigned j);
	//! Read-only index 2-d array at index i, j.
	const_reference at_index(unsigned i, unsigned j) const;

	//! Index 3-d array at index i, j, k.
	reference at_index(unsigned i, unsigned j, unsigned k);
	//! Read-only index 3-d array at index i, j, k.
	const_reference at_index(unsigned i, unsigned j, unsigned k) const;

public:
	//! Class destructor.
	virtual ~basic_array();

	//! Assignment operator.
	basic_array &operator=(const basic_array &rhs);
	//! Move operator.
	basic_array &operator=(basic_array &&rhs);

	//! Release this array's data.
	virtual void release();

	//! Get reference counter value pointer.
	refcount_type *refcounter() const {return this->_refcount;}
	//! Get array data block.
	pointer data() const {return this->_data;}
	//! Get raw pointer to the beginning of the array data.
	pointer data_begin() const {return this->_begin;}

	//! Get shape of the array.
	const index_array &shape() const {return this->_shape;}
	//! Get strides of the array.
	const index_array &strides() const {return this->_strides;}
	//! Get the dimension of the array.
	size_type dim() const {return this->_shape.size();}

	//! Check if array contains any data.
	bool is_valid() const;
	//! Check if this array has been constructed from continuous memory chunk.
	bool is_contiguous() const;
	//! Check if array is empty.
	bool empty() const;
	//! Get size of the array.
	size_type length() const;

	//! Check if array contains any data.
	operator bool() const {return static_cast<bool>(this->_data);}
};

///////////////////////////////////////////////////////////////////////////////
// Implementation
//

template<class _Tp>
void basic_array<_Tp>::allocate(const index_array &shape, unsigned byte_size) {
	auto p = shape.product();
	ASSERT(p);
	auto d = shape.size();
	this->_data = reinterpret_cast<pointer>(::operator new(p * sizeof(value_type)*byte_size));
	this->_begin = this->_data;
	this->_shape = shape;
	this->_strides.resize(d);
	this->_strides[d - 1] = byte_size;
	for (unsigned i = d - 1; i-- > 0; ) {
		this->_strides[i] = this->_shape[i + 1]*this->_strides[i + 1];
	}
}

template<class _Tp>
void basic_array<_Tp>::deallocate() {
	if (this->_refcount) { // else means the array memory is already empty or borrowed
		REF_DECREMENT(this->_refcount);
		if (!REF_CHECK(this->_refcount)) {
			::operator delete(this->_data); // then deallocate the memory.
			delete this->_refcount; // also release the reference counter.
		}
	}
	this->_data = nullptr;
	this->_refcount = nullptr;
	this->_shape.clear();
	this->_strides.clear();
}

template<class _Tp>
void basic_array<_Tp>::copy(const basic_array<_Tp> &cpy, bool deepCopy) {

	this->deallocate();

	if (deepCopy) {
		this->_refcount = REF_NEW;
		this->allocate(cpy.shape());

		if (cpy.is_contiguous()) {
			std::memcpy (this->_data, cpy._data, cpy.shape().product() * sizeof(_Tp));
		} else {
			switch( cpy.dim() ) {
				case 1:
					for (unsigned i = 0; i < cpy.shape()[0]; ++i) {
						*(this->_begin + i) = *(cpy._begin + i*(cpy._strides[0]));
					}
					break;
				case 2:
					for (unsigned i = 0; i < cpy.shape()[0]; ++i) {
						for (unsigned j = 0; j < cpy.shape()[1]; ++j) {
							*(this->_begin + i*this->_strides[0] + j) = *(cpy._begin + i*cpy._strides[0] + j*cpy._strides[1]);
						}
					}
					break;
				case 3:
					for (unsigned i = 0; i < cpy.shape()[0]; ++i) {
						for (unsigned j = 0; j < cpy.shape()[1]; ++j) {
							for (unsigned k = 0; k < cpy._shape[2]; ++k) {
								*(this->_begin + i*this->_strides[0] + j*this->_strides[1] + k) = 
									*(cpy._begin + i*cpy._strides[0] + j*cpy._strides[1] + k*cpy._strides[2]);
							}
						}
					}
					break;
				default:
					throw std::runtime_error("Not supported array shape, maximal supported dimension of the array is 3");
			}
		}
	} else {
		this->_data = cpy._data;
		this->_begin = cpy._begin;
		this->_refcount = cpy._refcount;
		this->_shape = cpy._shape;
		this->_strides = cpy._strides;

		if (this->_refcount) // may be invalid if array is null or with borrowed data.
			REF_INCREMENT(this->_refcount);
	}
}

template<class _Tp>
void basic_array<_Tp>::move(basic_array<_Tp> &&move) {
	this->_data = move._data;
	this->_begin = move._begin;
	this->_shape = move._shape;
	this->_strides = move._strides;
	this->_refcount = move._refcount;
	move._data = nullptr;
	move._begin = nullptr;
	move._shape.clear();
	move._strides.clear();
	move._refcount = nullptr;
}

template<class _Tp>
basic_array<_Tp>::basic_array() :
	_data(0), _begin(0), _refcount(0) 
{
}

template<class _Tp>
basic_array<_Tp>::basic_array(pointer data, pointer begin, const index_array &shape, const index_array &strides, refcount_type *refcount):
_data(data), _begin(begin), _shape(shape), _strides(strides), _refcount(refcount) {
	ASSERT(data && begin && refcount);
	REF_INCREMENT(refcount);
}

template<class _Tp>
basic_array<_Tp>::basic_array(size_type size):
_data(0), _begin(0), _refcount(REF_NEW) {
	this->allocate({size});
}

template<class _Tp>
basic_array<_Tp>::basic_array(size_type size1, size_type size2) :
	_data(0), _begin(0), _refcount(REF_NEW) {
	this->allocate({size1, size2});
}

template<class _Tp>
basic_array<_Tp>::basic_array(size_type size1, size_type size2, size_type size3):
	_data(0), _begin(0), _refcount(REF_NEW) {
	this->allocate({size1, size2, size3});
}

template<class _Tp>
basic_array<_Tp>::basic_array(const basic_array &cpy, bool deepCopy):
_refcount(nullptr), _data(nullptr) {
	this->copy(cpy, deepCopy);
}

template<class _Tp>
basic_array<_Tp>::basic_array(basic_array &&move):
_refcount(move._refcount), _data(move._data) {
	this->move(std::forward<basic_array<_Tp> >(move));
}

template<class _Tp>
basic_array<_Tp>::~basic_array() {
	this->deallocate();
}

template<class _Tp>
basic_array<_Tp> &basic_array<_Tp>::operator=(const basic_array<_Tp> &rhs) {
	if (this != &rhs) {
		this->copy(rhs, false);
	}
	return *this;
}

template<class _Tp>
basic_array<_Tp> &basic_array<_Tp>::operator=(basic_array<_Tp> &&rhs) {
	if (this != &rhs) {
		this->move(std::forward<basic_array<_Tp> >(rhs));
	}
	return *this;
}

template<class _Tp>
bool basic_array<_Tp>::is_valid() const {
	return static_cast<bool>(this->_data);
}

template<class _Tp>
bool basic_array<_Tp>::is_contiguous() const {
	ASSERT(this->_data);
	if (this->_shape.size() == 1)
		return this->_strides[0] == 1;
	else {
		if (this->_strides[this->_strides.size() - 1] != 1)
			return false;
		else {
			for (unsigned i = 0; i < this->_shape.size() - 1; ++i) {
				if ( this->_strides[i] != this->_shape[i + 1]*this->_strides[i + 1] )
					return false;
			}
		}	
	}
	return true;
}

template<class _Tp>
bool basic_array<_Tp>::empty() const {
	return this->_data == nullptr;
}

template<class _Tp>
typename basic_array<_Tp>::size_type basic_array<_Tp>::length() const {
	return this->_shape.product();
}

template<class _Tp>
void basic_array<_Tp>::release() {
	this->deallocate();
}

template<class _Tp>
typename basic_array<_Tp>::reference basic_array<_Tp>::at_index(unsigned i) {
	ASSERT(this->dim() == 1 && i < this->_shape[0]);
	return *(this->_begin + i*this->_strides[0]);
}

template<class _Tp>
typename basic_array<_Tp>::const_reference basic_array<_Tp>::at_index(unsigned i) const {
	ASSERT(this->dim() == 1 && i < this->_shape[0]);
	return *(this->_begin + i*this->_strides[0]);
}

template<class _Tp>
typename basic_array<_Tp>::reference basic_array<_Tp>::at_index(unsigned i, unsigned j) {
	ASSERT(this->dim() == 2 && i < this->_shape[0] && j < this->_shape[1]);
	return *(this->_begin + i*this->_strides[0] + j*this->_strides[1]);
}

template<class _Tp>
typename basic_array<_Tp>::const_reference basic_array<_Tp>::at_index(unsigned i, unsigned j) const {
	ASSERT(this->dim() == 2 && i < this->_shape[0] && j < this->_shape[1]);
	return *(this->_begin + i*this->_strides[0] + j*this->_strides[1]);
}

template<class _Tp>
typename basic_array<_Tp>::reference basic_array<_Tp>::at_index(unsigned i, unsigned j, unsigned k) {
	ASSERT(this->dim() == 3 && i < this->_shape[0] && j < this->_shape[1] && k < this->_shape[2]);
	return *(this->_begin + i*this->_strides[0] + j*this->_strides[1] + k*this->_strides[2]);
}

template<class _Tp>
typename basic_array<_Tp>::const_reference basic_array<_Tp>::at_index(unsigned i, unsigned j, unsigned k) const {
	ASSERT(this->dim() == 3 && i < this->_shape[0] && j < this->_shape[1] && k < this->_shape[2]);
	return *(this->_begin + i*this->_strides[0] + j*this->_strides[1] + k*this->_strides[2]);
}

}
#endif /* end of include guard: ARRAY_HPP_MVWJDTQ1 */
