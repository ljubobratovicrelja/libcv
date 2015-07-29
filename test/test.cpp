//The MIT License (MIT)
//
//Copyright (c) 2015 Relja Ljubobratovic, ljubobratovic.relja@gmail.com
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.
//
// Description:
// Testing program for libcv. Contains assertion checks for various algorithms, but also
// performs print-outs in which used can recognize if printed result is correct.
// Also shows images if libcv is compiled with gui support.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#include <iostream>
#include <algorithm>

#include "../include/vector.hpp"
#include "../include/matrix.hpp"
#include "../include/image.hpp"
#include "../include/io.hpp"
#include "../include/improc.hpp"
#include "../include/gui.hpp"

/*
	data type test check module
*/

void cv_vector_test() {
	cv::vector<float> vec;
	ASSERT(vec.empty());

	unsigned vec_size = 6;

	vec.create(vec_size);
	ASSERT(vec.length() == vec_size);
	ASSERT(vec);

	vec[0] = 1;
	vec[1] = 3;
	vec[2] = 5;
	vec[3] = 7;
	vec[4] = 9;
	vec[5] = 11;

	auto m = vec.min();
	ASSERT(*m == 1);

	m = vec.max();
	ASSERT(*m == vec[vec_size - 1]);

	auto st_vec = vec(0, vec_size - 1, 2);
	ASSERT(st_vec[0] == vec[0] && st_vec[1] == vec[2] && st_vec[2] == vec[4]);
	ASSERT(&st_vec[0] == &vec[0] && &st_vec[1] == &vec[2] && &st_vec[2] == &vec[4]);

	const cv::vector<float> const_st_vec = vec(0, vec_size - 1, 2);
	ASSERT(const_st_vec[0] == vec[0] && const_st_vec[1] == vec[2] && const_st_vec[2] == vec[4]);

	cv::vector<int> init_vec = {1, 3, 4};
	ASSERT(init_vec.length() == 3 && init_vec[0] == 1 && init_vec[1] == 3 && init_vec[2] == 4);

	auto vec_cpy = st_vec.clone();
	ASSERT(std::equal(vec_cpy.begin(), vec_cpy.end(), st_vec.begin()));
	ASSERT(&vec_cpy[0] != &st_vec[0]);

	// distance ------------------------------
	
	cv::vector<int> a = {1, 2};
	cv::vector<int> b = {1, 3};

	ASSERT(a.distance(b) == 1.);

	cv::vector<float> a_f = a; // conversion to float.
	a_f.normalize();

	auto a_norm = a.norm(); // L2 by default

	for (unsigned i = 0; i < a.length(); ++i) {
		ASSERT(a_f[i] == static_cast<float>((a[i]) / a_norm));
	}
}

void cv_matrix_test() {

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
}

void cv_image_array_test() {

	cv::image_array im = cv::imread("/home/relja/Lenna.png", cv::UINT8, 1);
	cv::matrixr im_r = im.as_type(cv::REAL);

	cv::matrixr sobel_x = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
	cv::matrixr sobel_y = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};

	cv::matrixr gauss_kernel = cv::gauss({5, 5});
	gauss_kernel /= cv::norm(gauss_kernel, cv::Norm::L1);

	std::cout << gauss_kernel << std::endl;

	cv::matrixr f_x, f_y, blur;

	f_x = cv::conv(im_r, sobel_x);
	f_y = cv::conv(im_r, sobel_y);

	auto f_grad = f_x + f_y;

	cv::image_array im_grad = cv::normalize(f_grad);
	im_grad.convert_to(cv::UINT8);

	blur = cv::conv(im_r, gauss_kernel);

	cv::image_array im_blur = blur;
	im_blur.convert_to(cv::UINT8);

	auto im_rgb = cv::imread("/home/relja/Lenna.png");
	auto ch_rgb = im_rgb.split();

	im_rgb.merge(ch_rgb);

#ifndef CV_IGNORE_GUI
	cv::imshow("lena gray", im);
	cv::imshow("lena color", im_rgb);
	cv::imshow("lenas edges", im_grad);
	cv::imshow("lena blurred", im_blur);
	std::cout << "Press any key to proceed..." << std::endl;
	cv::wait_key();
#endif
}

int main() {

	cv_vector_test();
	cv_matrix_test();
	cv_image_array_test();

	std::cout << "libcv test passed!" << std::endl;

	return EXIT_SUCCESS;
}
