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

int lmdif(optimization_fcn fcn, int m, int n, double *x, int maxfev, double ftol, double xtol, double gtol, double epsfcn) {

	ASSERT(m > n && x);

	double* fvec=new double[m]; //no need to populate

	int mode=1; //some internal thing
	double factor=1; // a default recommended value
	int nprint=0; //don't know what it does
	int info=0; //output variable
	int nfev=0; //output variable will store no. of function evals

	double* diag=new double[n]; //some internal thing
	double* fjac=new double[m*n]; //output array of jacobian

	int ldfjac=m; //recommended setting
	int* ipvt=new int[n]; //for internal use

	double* qtf=new double[n]; //for internal use
	double* wa1=new double[n]; //for internal use
	double* wa2=new double[n]; //for internal use
	double* wa3=new double[n]; //for internal use
	double* wa4=new double[m]; //for internal use

	lmdif_(fcn, &m, &n, x, fvec,  &ftol,
	       &xtol, &gtol, &maxfev, &epsfcn, diag,  &mode,  &factor,
	       &nprint,  &info, &nfev, fjac, &ldfjac, ipvt, qtf,
	       wa1, wa2, wa3, wa4);

	return info;
}

int lmdif1(optimization_fcn fcn, int m, int n, double *x, double tol) {
	ASSERT(m > n && x && tol > std::numeric_limits<double>::min());

	int info = 0;

	auto iwa = new int[n];
	int lwa = (m*n)+(5*n)+m+10;
	auto wa = new double[lwa];
	auto fvec = new double[m];

	lmdif1_(fcn, &m, &n, x, fvec, &tol, &info, iwa, wa, &lwa);

	delete [] iwa;
	delete [] wa;
	delete [] fvec;

	return info;
}

}

