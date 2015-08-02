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
// 2D Region structure implementation.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef REGION_HPP_JXUFAR0O
#define REGION_HPP_JXUFAR0O

#include "fwd.hpp"

#include <iostream>

namespace cv {

template<typename _Tp>
class region {
  public:
	_Tp x; //!< X coordinate of the region top-left corner.
	_Tp y; //!< Y coordinate of the region top-left corner.
	_Tp width;  //!< Width of the region.
	_Tp height; //!< Height of the region.

	//! Default constructor.
	region();
	//! Construction using region values.
	region(_Tp x, _Tp y, _Tp width, _Tp height);
	//! Class destructor.
	~region();

	// operators
	bool operator==(const region& b)const ;
	bool operator!=(const region& b) const ;
	bool operator<=(const region& b) const ;
	bool operator>=(const region& b) const ;
	bool operator>(const region& b) const ;
	bool operator<(const region& b) const ;

	friend std::ostream& operator<<(std::ostream& stream, const region& reg) {
		stream << reg.x << ", " << reg.y << " : " << reg.width << ", "
		       << reg.height;
		return stream;
	}
};

typedef region<byte> regionb; //!< Byte(unsigned char) typed 2D region.
typedef region<int> regioni; //!< Integer typed 2D region.
typedef region<float> regionf; //!< Floating point single precision typed 2D region.
typedef region<double> regiond; //!< Floating point double precision typed 2D region.
typedef region<long long> regionl; //!< Long long typed 2D region.

template<typename _Tp>
region<_Tp>::region() :
	x(0), y(0), width(0.0), height(0.0) {
}

template<typename _Tp>
region<_Tp>::region(_Tp x, _Tp y, _Tp width, _Tp height) :
	x(x), y(y), width(width), height(height) {
}

template<typename _Tp>
region<_Tp>::~region() {
}

template<typename _Tp>
bool region<_Tp>::operator==(const region& b) const {
	if ((*this).width == b.width && (*this).height == b.height
	        && (*this).x == b.x && (*this).y == b.y) {
		return true;
	} else {
		return false;
	}
}

template<typename _Tp>
bool region<_Tp>::operator!=(const region& b)const  {
	if ((*this) == b) {
		return false;
	} else {
		return true;
	}
}

template<typename _Tp>
bool region<_Tp>::operator<=(const region& b) const {
	if ((*this).width <= b.width && (*this).height <= b.height) {
		return true;
	} else {
		return false;
	}
}

template<typename _Tp>
bool region<_Tp>::operator>=(const region& b) const {
	if ((*this).width >= b.width && (*this).height >= b.height) {
		return true;
	} else {
		return false;
	}
}

template<typename _Tp>
bool region<_Tp>::operator>(const region& b)const  {
	if ((*this).width > b.width && (*this).height > b.height) {
		return true;
	} else {
		return false;
	}
}

template<typename _Tp>
bool region<_Tp>::operator<(const region<_Tp>& b) const {
	if ((*this).width < b.width && (*this).height < b.height) {
		return true;
	} else {
		return false;
	}
}

}

#endif /* end of include guard: REGION_HPP_JXUFAR0O */

