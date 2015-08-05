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
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#include "../include/math.hpp"

#include <limits>
#include <cmath>

namespace cv {

namespace internal {
template<typename _Tp>
int sgn(_Tp d, bool useEpsilon) {
	auto eps = std::numeric_limits<_Tp>::epsilon();
	if (useEpsilon)
		return (d < -eps ? -1 : d > eps);
	else {
		if (d == 0) {
			return 0;
		} else {
			return (d < 0 ? -1 : 1);
		}
	}
}
}

int sgn(int d) {
	return internal::sgn(d, false);
}

int sgn(real_t d) {
	return internal::sgn(d, true);
}

int sgn(long long d) {
	return internal::sgn(d, false);
}

int sgn(byte d) {
	return internal::sgn(d, false);
}

real_t quantize(real_t val, real_t max, real_t stepSize) {
	return (val / max) * stepSize;
}

void quadratic_solve(real_t a, real_t b, real_t c, real_t &x1, real_t &x2) {

	real_t sqrtVal = pow(b, 2) - (4 * a * c);

	x1 = (-1 * b / 2) + (sqrt(sqrtVal) / 2);
	x2 = (-1 * b / 2) - (sqrt(sqrtVal) / 2);
}

real_t eps(real_t val) {
	return (val - std::nextafter(val, std::numeric_limits<real_t>::epsilon()));
}

real_t derive(real_t (*f)(real_t), real_t x, real_t dx) {
	return ((f(x + dx) - f(x)) / dx);
}

real_t trunc_real(real_t d) {
    return (d>0) ? floor(d) : ceil(d) ;
}

bool cmp_real(real_t a , real_t b) {
    return static_cast<bool>((trunc_real(1000.0 * a) == trunc_real(1000.0 * b)));
}

}

