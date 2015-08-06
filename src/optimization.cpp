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

#include "../include/optimization.hpp"

#include <minpack.h>

#include <limits>


namespace cv {

int lmdif1(optimization_fcn fcn, int m, int n, double *x, double tol) {
	ASSERT(m > n && x && tol > std::numeric_limits<double>::min());

	int info = 0;

	auto iwa = new int[n];
	int lwa = (m*n)+(5*n)+m+10;
	auto wa = new double[lwa];
	auto fvec = new double[m];

	lmdif1_(fcn, &m, &n, x, fvec, &tol, &info, iwa, wa, &lwa);

	return info;
}

}
