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
// Example program for usage of cv::matrix


#include <iostream>
#include <algorithm>

#include "../include/matrix.hpp"


int main() {

	cv::matrix<double> mat(3, 3);
	std::cout << "3x3 matrix:\n" << mat << std::endl;

	mat.fill(3);

	std::cout << "3x3 matrix filled with 3:\n" << mat << std::endl;

	mat.reshape(1, 9);

	std::cout << "1x9 reshaped matrix:\n" << mat << std::endl;

	mat.reshape(3, 3);

	std::cout << "3x3 reshaped to original:\n" << mat << std::endl;

	auto r = mat.row(1);
	r.fill(0);
	ASSERT(mat(1, 0) == 0 && mat(1, 1) == 0 && mat(1, 2) == 0);

	auto c = mat.col(1);
	c.fill(1);
	ASSERT(mat(0, 1) == 1 && mat(1, 1) == 1 && mat(2, 1) == 1);

	cv::matrix3f mat_3f(3, 3);
	mat_3f.fill({255, 15, 354});
	std::cout << mat_3f << std::endl;

	cv::vectori v = {1, 2, 3, 4};
	cv::matrixi mat_from_vector = v;

	ASSERT(v.data() ==	mat_from_vector.data());
	ASSERT(mat_from_vector.rows() == 1 && mat_from_vector.cols() == v.length());

	std::cout << "Mat from vector:\n" << mat_from_vector << std::endl;

	cv::vec3i vx = {1, 2, 3};
	cv::matrixi mat_from_vectorx = vx;

	std::cout << mat_from_vectorx.size() << std::endl;

	ASSERT(mat_from_vectorx.rows() == 1 && mat_from_vectorx.cols() == 3);

	std::cout << "Mat from vectorx:\n" << mat_from_vectorx << std::endl;

	return EXIT_SUCCESS;
}

