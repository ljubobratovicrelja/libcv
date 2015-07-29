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
// Iterator classes implementation.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef ITERATOR_HPP_1IXAVC6G
#define ITERATOR_HPP_1IXAVC6G


#include <iostream>
#include <iterator>

#include "fwd.hpp"


namespace cv {

 template<class _Tp>
class bidirectional_iterator: public std::iterator < std::bidirectional_iterator_tag, _Tp > {
public:
	typedef typename std::iterator<std::bidirectional_iterator_tag, _Tp>::value_type value_type;
	typedef typename std::iterator<std::bidirectional_iterator_tag, _Tp>::difference_type difference_type;
	typedef typename std::iterator<std::bidirectional_iterator_tag, _Tp>::pointer pointer;
	typedef typename std::iterator<std::bidirectional_iterator_tag, _Tp>::reference reference;
	typedef typename std::iterator<std::bidirectional_iterator_tag, _Tp>::iterator_category iterator_category;
protected:
	_Tp* _ptr;
	unsigned _stride;
public:

	bidirectional_iterator();
	bidirectional_iterator(pointer ptr, unsigned stride = 1);
	bidirectional_iterator(const bidirectional_iterator<_Tp>& cpy);
	virtual ~bidirectional_iterator();

	bidirectional_iterator<_Tp>& operator=(const bidirectional_iterator<_Tp>& rhs);

	_Tp& operator*();
	const _Tp& operator*() const;
	_Tp* operator->();
	const _Tp* operator->() const;
	_Tp* getPtr() const;
	const _Tp* getConstPtr() const;
	unsigned stride() const;

	bool operator==(const bidirectional_iterator<_Tp>& rawIterator) const;
	bool operator!=(const bidirectional_iterator<_Tp>& rawIterator) const;

	bidirectional_iterator<_Tp>& operator++();
	bidirectional_iterator<_Tp>& operator--();
	bidirectional_iterator<_Tp> operator++(int);
	bidirectional_iterator<_Tp> operator--(int);

};

 template<class _Tp>
class random_access_iterator: public std::iterator < std::random_access_iterator_tag, _Tp > {
public:
	typedef typename std::iterator<std::random_access_iterator_tag, _Tp>::value_type value_type;
	typedef typename std::iterator<std::random_access_iterator_tag, _Tp>::difference_type difference_type;
	typedef typename std::iterator<std::random_access_iterator_tag, _Tp>::pointer pointer;
	typedef typename std::iterator<std::random_access_iterator_tag, _Tp>::reference reference;
	typedef typename std::iterator<std::random_access_iterator_tag, _Tp>::iterator_category iterator_category;
private:
	_Tp* _ptr;
public:
	random_access_iterator(_Tp* ptr = nullptr) {
		_ptr = ptr;
	}
	random_access_iterator(const random_access_iterator<_Tp>& rawIterator) = default;
	virtual ~random_access_iterator() {
	}

	random_access_iterator<_Tp>& operator=(const random_access_iterator<_Tp>& rawIterator) = default;
	random_access_iterator<_Tp>& operator=(_Tp* ptr) {
		_ptr = ptr;
		return (*this);
	}

	_Tp &operator[](difference_type off) const {
		return _ptr[off];
	}

	_Tp& operator*() {
		return *_ptr;
	}
	const _Tp& operator*() const {
		return *_ptr;
	}
	_Tp* operator->() {
		return _ptr;
	}

	virtual const _Tp* operator->() const {
		return _ptr;
	}

	_Tp* getPtr() const {
		return _ptr;
	}
	const _Tp* getConstPtr() const {
		return _ptr;
	}

	bool operator==(const random_access_iterator<_Tp>& rawIterator) const {
		return (_ptr == rawIterator.getConstPtr());
	}

	bool operator!=(const random_access_iterator<_Tp>& rawIterator) const {
		return (_ptr != rawIterator.getConstPtr());
	}

	random_access_iterator<_Tp>& operator+=(difference_type movement) {
		_ptr += movement;
		return (*this);
	}

	random_access_iterator<_Tp>& operator-=(difference_type movement) {
		_ptr -= movement;
		return (*this);
	}

	random_access_iterator<_Tp>& operator++() {
		_ptr++;
		return (*this);
	}

	random_access_iterator<_Tp>& operator--() {
		_ptr--;
		return (*this);
	}

	random_access_iterator<_Tp> operator++(int) {
		auto temp(*this);
		_ptr++;
		return temp;
	}

	random_access_iterator<_Tp> operator--(int) {
		auto temp(*this);
		_ptr--;
		return temp;
	}

	random_access_iterator<_Tp> operator+(difference_type movement) const {
		return random_access_iterator<_Tp>(_ptr + movement);

	}

	random_access_iterator<_Tp> operator-(difference_type movement) const {
		return random_access_iterator<_Tp>(_ptr - movement);
	}

	friend random_access_iterator<_Tp> operator+(difference_type movement,
		const random_access_iterator<_Tp> &iter) {
		return random_access_iterator<_Tp>(movement + iter._ptr);

	}

	friend random_access_iterator<_Tp> operator-(difference_type movement,
		const random_access_iterator<_Tp> &iter) {
		return random_access_iterator<_Tp>(movement - iter._ptr);
	}

	bool operator>(const random_access_iterator<_Tp> &rhs) const {
		return (this->_ptr > rhs._ptr);
	}

	bool operator<(const random_access_iterator<_Tp> &rhs) const {
		return (this->_ptr < rhs._ptr);
	}

	bool operator>=(const random_access_iterator<_Tp> &rhs) const {
		return (this->_ptr >= rhs._ptr);
	}

	bool operator<=(const random_access_iterator<_Tp> &rhs) const {
		return (this->_ptr <= rhs._ptr);
	}

	std::ptrdiff_t operator-(const random_access_iterator<_Tp>& rawIterator) {
		return std::distance(rawIterator.getPtr(), this->getPtr());
	}
};


template<typename _Tp>
bidirectional_iterator<_Tp>::bidirectional_iterator() : _ptr(0), _stride(0) {

}

template<typename _Tp>
bidirectional_iterator<_Tp>::bidirectional_iterator(pointer ptr, unsigned stride) {
	this->_ptr = ptr;
	this->_stride = stride;
}

template<typename _Tp>
bidirectional_iterator<_Tp>::bidirectional_iterator(const bidirectional_iterator<_Tp>& cpy) : _ptr(cpy._ptr), _stride(cpy._stride) {

}

template<typename _Tp>
bidirectional_iterator<_Tp>::~bidirectional_iterator() {
}

template<typename _Tp>
bidirectional_iterator<_Tp>& bidirectional_iterator<_Tp>::operator=(const bidirectional_iterator<_Tp>& rhs) {
	if (this != &rhs) {
		this->_ptr = rhs._ptr;
		this->_stride = rhs._stride;
	}
}

template<typename _Tp>
_Tp& bidirectional_iterator<_Tp>::operator*() {
	return *_ptr;
}

template<typename _Tp>
const _Tp& bidirectional_iterator<_Tp>::operator*() const {
	return *_ptr;
}

template<typename _Tp>
_Tp* bidirectional_iterator<_Tp>::operator->() {
	return _ptr;
}

template<typename _Tp>
const _Tp* bidirectional_iterator<_Tp>::operator->() const {
	return _ptr;
}

template<typename _Tp>
_Tp* bidirectional_iterator<_Tp>::getPtr() const {
	return _ptr;
}

template<typename _Tp>
const _Tp* bidirectional_iterator<_Tp>::getConstPtr() const {
	return _ptr;
}

template<typename _Tp>
unsigned bidirectional_iterator<_Tp>::stride() const {
	return this->_stride;
}

template<typename _Tp>
bool bidirectional_iterator<_Tp>::operator==(const bidirectional_iterator<_Tp>& rawIterator) const {
	return (_ptr == rawIterator.getConstPtr());
}

template<typename _Tp>
bool bidirectional_iterator<_Tp>::operator!=(const bidirectional_iterator<_Tp>& rawIterator) const {
	return (_ptr != rawIterator.getConstPtr());
}

template<typename _Tp>
bidirectional_iterator<_Tp>& bidirectional_iterator<_Tp>::operator++() {
	_ptr += _stride;
	return (*this);
}

template<typename _Tp>
bidirectional_iterator<_Tp>& bidirectional_iterator<_Tp>::operator--() {
	_ptr -= _stride;
	return (*this);
}

template<typename _Tp>
bidirectional_iterator<_Tp> bidirectional_iterator<_Tp>::operator++(int) {
	auto temp(*this);
	_ptr += _stride;
	return temp;
}

template<typename _Tp>
bidirectional_iterator<_Tp> bidirectional_iterator<_Tp>::operator--(int) {
	auto temp(*this);
	_ptr -= _stride;
	return temp;
}


}

#endif /* end of include guard: ITERATOR_HPP_1IXAVC6G */
