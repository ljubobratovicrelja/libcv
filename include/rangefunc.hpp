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
// Collection of algorithms for range based usage. 
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef RANGEFUNC_HPP_8CYJB2QK
#define RANGEFUNC_HPP_8CYJB2QK

#include "array.hpp"
#include "vector.hpp"

namespace cv {

template<class iterator>
real_t sum(iterator begin, iterator end) {
	real_t s = 0.0;
	do s += *begin; while (++begin != end);
	return s;
}

template<class iterator>
real_t mean(iterator begin, iterator end) {
	return sum(begin, end) / static_cast<real_t>(std::distance(begin, end));
}

template <typename const_iterator>
real_t norm(const_iterator begin, const_iterator end, Norm ntype) {
	real_t nv = 0;
	switch (ntype) {
		case Norm::L1:
			do nv += *begin; while (++begin != end);
			break;
		case Norm::L2:
			do nv += pow(*begin, 2); while (++begin != end);
			nv = sqrtf(nv);
			break;
		default:
			throw std::runtime_error("Unsupported norm type - should be L1 - or L2");
	}
	return nv;
}

template<typename iterator>
void normalize(iterator begin, iterator end, Norm ntype) {
	auto nv = norm(begin, end, ntype);
	do *begin /= nv; while (++begin != end);
}

}

#endif /* end of include guard: RANGEFUNC_HPP_8CYJB2QK */
