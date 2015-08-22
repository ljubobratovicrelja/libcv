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
// Example program for usage of some of linear algebra solver functions from libcv.


#include <iostream>
#include <algorithm>

#include "../include/linalg.hpp"
#include "../include/matrix.hpp"


int main() {

	// inverse matrix calculation.
	cv::matrixr mat = {{32, 4, 5}, {1, 32, 4}, {31, 54, 54}};

	auto inv = mat.clone();

	cv::invert(inv);

	auto one_mat = mat * inv;

	std::cout << "Identity matrix? : \n" << one_mat << std::endl;

	// lu decomposition 
	cv::matrixr L, U, P;

	mat = {{7, 3, -11}, {-6, 7, 10}, {-11, 2, -2}};

	cv::lu_decomp(mat, L, U, P);

	auto comp_res = (P*L*U);

	// check if those are the same.
	ASSERT(std::memcmp((void*)mat.data(), (void*)comp_res.data(), 9*sizeof(real_t)));


	// svd decomposition 
	cv::matrixr S, Vt;

	cv::svd_decomp(mat, U, S, Vt);

	comp_res = (U*S*Vt);
	ASSERT(std::memcmp((void*)mat.data(), (void*)comp_res.data(), 9*sizeof(real_t)));

	// system solving : LU

	cv::matrixr a = {
		{6.80 , -6.05 , -0.45 ,  8.32 , -9.67},
		{-2.11,  -3.30,   2.58,   2.71,  -5.14},
		{5.66 ,  5.36 , -2.70 ,  4.35 , -7.26},
		{5.97 , -4.44 ,  0.27 , -7.17 ,  6.08},
		{8.23 ,  1.08 ,  9.04 ,  2.14 , -6.87}
	};

	cv::matrixr b = {
		{4.02 , -1.56 ,  9.81},
		{6.19 ,  4.00 , -4.09},
		{-8.22,  -8.67,  -4.57},
		{-7.57,   1.75,  -8.61},
		{-3.03,   2.86,   8.99}
	};

	cv::matrixr correct_solution = {
		{-0.800714, -0.389621, 0.955465 },
		{-0.695243, -0.554427, 0.22066 },
		{0.593915 ,0.842227, 1.90064 },
		{1.32173 ,-0.103802 ,5.35766 },
		{0.565756 ,0.105711 ,4.0406 }
	};

	cv::matrixr x;

	cv::lu_solve(a, b, x);

	std::cout << correct_solution << std::endl;
	std::cout << x << std::endl;

	return EXIT_SUCCESS;
}

