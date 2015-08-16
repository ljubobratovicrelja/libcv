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

#include "../include/cminpack.h"

#include <limits>

namespace cv {

int lmdif(optimization_fcn fcn, int m, int n, double *x, int maxfev, double ftol, double xtol, double gtol, double epsfcn) {

	ASSERT(m > n && x);

	double* fvec=new double[m]; 

	int mode=1;
	double factor=1;
	int info=0; 
	int nfev=0; 

	double* diag=new double[n]; 
	double** fjac=new double*[m]; 

	for (int i = 0; i < n; ++i) { fjac[i] = new double[n]; }

	int* ipvt=new int[n]; 

	double* qtf=new double[n]; 
	int *msk = new int[n];
	double* wa1=new double[n]; 
	double* wa2=new double[n]; 
	double* wa3=new double[n]; 
	double* wa4=new double[m]; 

	::lmdif(fcn, m, n, x, msk, fvec, ftol, xtol, gtol, maxfev, epsfcn, diag, mode, factor, &info, &nfev,
			fjac, ipvt, qtf, wa1, wa2, wa3, wa4);

	for (int i = 0; i < n; ++i) { delete [] fjac[i]; }

	delete [] fvec;
	delete [] diag;
	delete [] msk;
	delete [] fjac;
	delete [] qtf;
	delete [] wa1;
	delete [] wa2;
	delete [] wa3;
	delete [] wa4;

	return info;
}

int lmdif1(optimization_fcn fcn, int m, int n, double *x, double tol) {
	ASSERT(m > n && x && tol > std::numeric_limits<double>::min());

	int info = 0;

	int     *msk = new int[n];
	double  *fvec = new double[m];
	int     nfev;

	::lmdif0(fcn, m,n,x,msk,fvec,tol,&info,&nfev);

	delete [] fvec;
	delete [] msk;

	return info;
}

}


