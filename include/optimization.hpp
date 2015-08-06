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
// Minpack wrapper for non-linear optmization methods.
// 
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef OPTIMIZATION_HPP_YPHLDAFR
#define OPTIMIZATION_HPP_YPHLDAFR


#include <minpack.h>

#include "matrix.hpp"
#include "vector.hpp"


namespace cv {

/*!
 * @brief Type of the function for optmization.
 *
 * @param m number of functions
 * @param n numer of variables, must not exceed m
 * @param x array of length n - on input contains initial estimate of the
 * solution vector, on output contains final estimate of the vector.
 * @param fvec functions evaluated at x.
 * @param iflag signal for optimization termination. Is 0 if optimization 
 * should terminate.
 */
typedef void (*optimization_fcn)(int *m, int *n, double *x, double *fvec, int *iflag);

/*!
 * @brief Wrapper structure for convinient use of minpacks lmdif1 routine.
 */
int lmdif1(optimization_fcn fcn, int m, int n, double *x, double tol = 10e-16);

}

#endif /* end of include guard: OPTIMIZATION_HPP_YPHLDAFR */

