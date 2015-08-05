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
// Various mathematical methods.
// 
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef MATH_HPP_CNMJRQQF
#define MATH_HPP_CNMJRQQF


#include "fwd.hpp"

#include <limits>

namespace cv {


//! Sign function.
int CV_EXPORT sgn(int d);
//! Sign function.
int CV_EXPORT sgn(long long d);
//! Sign function.
int CV_EXPORT sgn(byte d);
//! Sign function. Uses epsilon value for value error.
int CV_EXPORT sgn(real_t d);
//! Quantize value.
real_t CV_EXPORT quantize(real_t val, real_t max, real_t stepSize);
//! Solve quadratic equation.
void CV_EXPORT quadratic_solve(real_t a, real_t b, real_t c, real_t &x1, real_t &x2);
//! Epsilon value of a real_t.
real_t CV_EXPORT eps(real_t val);
//! Simple calculation of derivative of a function of a given value (f(x)).
real_t CV_EXPORT derive(real_t (*f)(real_t), real_t x, real_t dx = 1e-10);
//! Floating point (real number) comparison with x1000.0 error tolerance.
bool CV_EXPORT cmp_real(real_t a , real_t b);

// Templates ////////////////////////////////////////////////////////////////////////////////////////

//! Is the queried value close enough within the given error?
template<typename _Tp> bool is_aproximation(const _Tp &query, const _Tp &source, const _Tp &error = std::numeric_limits<_Tp>::epsilon()) {
	return (fabs(source - query) < error);
}

}

#endif /* end of include guard: MATH_HPP_CNMJRQQF */
