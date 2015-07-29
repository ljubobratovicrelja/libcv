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
// Module contains wrapper (template) class for atomic manipulation of objects
// using OpenMP atomic pragma. 
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef ATOMIC_HPP_8OD6ZLUE
#define ATOMIC_HPP_8OD6ZLUE


#include "fwd.hpp"


namespace cv {


/*!
@brief Wrapper class for objects that are modified with atomic lock.

Concept is adopted from std::atomic, but it's functionality
is defined for libcv defined paralellization package.
*/
template<typename _Tp>
class CV_EXPORT atomic {
public:
	typedef _Tp value_type;
	typedef _Tp *pointer;
	typedef _Tp &reference;
	typedef const _Tp &const_reference;

private:
	_Tp _object;
public:
	//! Class constructor.
	atomic() {

	}
	//! Construct from value.
	atomic(const_reference value):
		_object(value) {

	}
	//! Copy constructor.
	atomic(const atomic<_Tp> &cpy):
		_object(cpy._object) {

	}
	//! Move constructor.
	atomic(atomic &&move):
		_object(std::move(move._object)) {

	}
	//! Class destructor.
	~atomic() {

	}

	atomic &operator=(const atomic& rhs) {
		if (this != &rhs) {
			this->_object = rhs._object;
		}
		return *this;
	}
	atomic &operator=(const_reference rhs) {
		if (&this->_object != &rhs) {
			this->_object = rhs;
		}
		return *this;
	}

	atomic &operator=(atomic &&rhs) {
		if (this != &rhs) {
			this->_object = std::move(rhs._object);
		}
		return *this;
	}
	atomic &operator=(_Tp &&rhs) {
		if (&this->_object != &rhs) {
			this->_object = std::move(rhs);
		}
		return *this;
	}

	operator _Tp() const {
		return this->_object;
	}

	atomic &operator++() {
		OMP_ATOMIC
			this->_object++;

		return *this;
	}

	atomic &operator--() {
		OMP_ATOMIC
			this->_object--;

		return *this;
	}

	atomic operator++(int) {
		_Tp tmp = this->_object;

		OMP_ATOMIC
			this->_object++;

		return tmp;
	}

	atomic operator--(int) {
		_Tp tmp = this->_object;

		OMP_ATOMIC
			this->_object--;

		return tmp;
	}

	atomic &operator+=(const_reference rhs) {
		OMP_ATOMIC
			this->_object += rhs;

		return *this;
	}

	atomic &operator-=(const_reference rhs) {
		OMP_ATOMIC
			this->_object -= rhs;

		return *this;
	}

	atomic &operator*=(const_reference rhs) {
		OMP_ATOMIC
			this->_object *= rhs;

		return *this;
	}

	atomic &operator/=(const_reference rhs) {
		OMP_ATOMIC
			this->_object /= rhs;

		return *this;
	}


	atomic operator+(const_reference rhs) const {
		atomic obj(*this);
		obj += rhs;
		return obj;
	}

	atomic operator-(const_reference rhs) const {
		atomic obj(*this);
		obj -= rhs;
		return obj;
	}

	atomic operator*(const_reference rhs) const {
		atomic obj(*this);
		obj *= rhs;
		return obj;
	}

	atomic operator/(const_reference rhs) const {
		atomic obj(*this);
		obj /= rhs;
		return obj;
	}

	bool operator==(const atomic &rhs) const {
		return (this->_object == rhs._object);
	}

	bool operator!=(const atomic &rhs) const {
		return !(operator==(rhs));
	}

	bool operator<(const atomic &rhs) const {
		return (this->_object < rhs._object);
	}

	bool operator<=(const atomic &rhs) const {
		return (this->_object <= rhs._object);
	}

	bool operator>(const atomic &rhs) const {
		return (this->_object > rhs._object);
	}

	bool operator>=(const atomic &rhs) const {
		return (this->_object >= rhs._object);
	}

	friend std::ostream &operator << (std::ostream &stream, const atomic &rhs) {
		stream << rhs._object;
		return stream;
	}
};

}

#endif /* end of include guard: ATOMIC_HPP_8OD6ZLUE */
