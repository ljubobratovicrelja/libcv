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
	cv::matrixr mat = {{32.f, 4.f, 5.f}, {1.f, 32.f, 4.f}, {31.f, 54.f, 54.f}};

	auto inv = mat.clone();

	cv::invert(inv);

	auto one_mat = mat * inv;

	std::cout << "Identity matrix? : \n" << one_mat << std::endl;

	// lu decomposition 
	cv::matrixr L, U, P;

	mat = {{7.f, 3.f, -11.f}, {-6.f, 7.f, 10.f}, {-11.f, 2.f, -2.f}};

	cv::lu_decomp(mat, L, U, P);

	auto comp_res = (P*L*U);

	// check if those are the same.
	ASSERT(std::memcmp((void*)mat.data(), (void*)comp_res.data(), 9*sizeof(real_t)));


	// svd decomposition 
	cv::matrixr S, Vt;

	cv::svd_decomp(mat, U, S, Vt);

	comp_res = (U*S*Vt);
	ASSERT(std::memcmp((void*)mat.data(), (void*)comp_res.data(), 9*sizeof(real_t)));

	/*
	NOTE:
	Even if the real_t is double, initializer list
	has floats in it. On vc, narrowing down 
	from double to float produces an error, but 
	in gcc this is just a warning. 

	So for the sake of compatibility, you should
	always initialize real_t matrices with floats,
	and if you need doubles, compile libcv with 
	double as real - DOUBLE_REAL switch in CMake.
	*/

	// system solving : LU
	cv::matrixr a = {
		{6.80f, -6.05f, -0.45f, 8.32f, -9.67f},
		{-2.11f, -3.30f, 2.58f, 2.71f, -5.14f},
		{5.66f, 5.36f, -2.70f, 4.35f, -7.26f},
		{5.97f, -4.44f, 0.27f, -7.17f, 6.08f},
		{8.23f, 1.08f, 9.04f, 2.14f, -6.87f}
	};

	cv::matrixr b = {
		{4.02f , -1.56f,  9.81f},
		{6.19f,  4.00f, -4.09f},
		{-8.22f,  -8.67f,  -4.57f},
		{-7.57f,   1.75f,  -8.61f},
		{-3.03f,   2.86f,   8.99f}
	};

	cv::matrixr correct_solution = {
		{-0.800714f, -0.389621f, 0.955465f },
		{-0.695243f, -0.554427f, 0.22066f },
		{0.593915f, 0.842227f, 1.90064f },
		{1.32173f, -0.103802f, 5.35766f },
		{0.565756f, 0.105711f, 4.0406f }
	};

	cv::matrixr x;

	cv::lu_solve(a, b, x);

	std::cout << "Correct solution:\n" << correct_solution << std::endl;
	std::cout << "LU Solve calculated solution:\n" << x << std::endl;
	std::cout << "Error: " << cv::distance(correct_solution, x) << std::endl;

	return EXIT_SUCCESS;
}

