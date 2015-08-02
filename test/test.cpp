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
#include "../include/kdtree.hpp"
#include "../include/draw.hpp"

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
	im.convert_to(cv::REAL);

	cv::matrixr im_r, f_x, f_y, blur, f_grad;

	im_r = im;
	
	cv::calc_derivatives(im_r, f_x, f_y);

	f_grad = cv::normalize(f_x + f_y);

	cv::imshow("im", im);
	cv::imshow("gradients", f_grad);

	cv::wait_key();

	return;
}

void cv_gradient_test() {

	cv::image_array im = cv::imread("/home/relja/Lenna.png", cv::UINT8, 1);
	im.convert_to(cv::REAL);

	cv::matrixr im_r, f_x, f_y, blur, f_grad, h, s, non_max;

	im_r = im;

	cv::calc_derivatives(im_r, f_x, f_y);

	f_grad = cv::normalize(f_x + f_y);

	h = cv::normalize(cv::harris(im_r, 3, 0.8, 1));
	s = cv::normalize(cv::good_features(im_r, 3, 1));

	non_max = s.clone();
	cv::filter_non_maximum(non_max, 5);

	cv::imshow("im", im);
	cv::imshow("gradients", f_grad);
	cv::imshow("harris corners", h);
	cv::imshow("shi-tomasi corners", s);
	cv::imshow("shi-tomasi corners non-max", non_max);

	cv::wait_key();

	return;
}


void cv_kd_tree_test() {

	srand(time(NULL));

	std::vector<cv::vec2i> result;

	std::vector<cv::vec2i> source, results;
	std::vector<unsigned> idResults;

	for (unsigned i = 0; i < 1000; i++) {
		cv::vec2i ptr = { rand() % 599, rand() % 599 };
		source.push_back(ptr);
	}

	double radius = 250.0;
	int nnCount = 6;

	cv::vec2i ptr_search = { rand() % 600, rand() % 600 };

	cv::kd_tree2i kd(source);

	while (true) {

		if (nnCount < 3) {
			nnCount = 3;
		}

		cv::matrix3b img = cv::matrix3b::zeros(600,600);

		for (unsigned i = 0; i < source.size(); i++) {
			cv::draw_circle(img, source[i], 5, cv::vec3b{0, 0, 255});
		}

		kd.knn_index(ptr_search, nnCount, idResults, radius);

		cv::draw_circle(img, ptr_search, 5, {255, 0, 0});
		cv::draw_circle(img, ptr_search, radius, {255, 255, 0});

		for (auto id : idResults) {
				cv::draw_circle(img, source[id], 5, {0, 255, 0});
		}

		cv::imshow("KD", img);
		char key = cv::wait_key();

		switch (key) {
		case 'q':
			return;
		case 'o':
			radius += 10.0;
			break;

		case 'p':
			radius -= 10.0;
			break;
		case 'k':
			nnCount += 1;
			break;
		case 'l':
			nnCount -= 1;
			break;
		default:
			ptr_search = { rand() % 600, rand() % 600 };
			break;
		}
	}
}


void cv_draw_test() {

	cv::matrix3b img = cv::matrix3b::zeros(512, 512);

	cv::polygoni poly;
	poly << cv::vec2i{100, 300} << cv::vec2i{150, 200} << cv::vec2i{200, 300};

	cv::draw_polygon(img, poly, {255, 0, 0}, 1, true);

	//cv::draw_point(img, {256, 256}, {255, 0, 0}, 1);
	//cv::draw_line(img, {200, 200}, {300, 250}, {0, 255, 0});
	//cv::draw_circle(img, {256, 256}, 15, {0, 0, 255});

	cv::imshow("draw image", img);
	cv::wait_key();

}


int main() {

	//cv_vector_test();
	//cv_matrix_test();
	//cv_image_array_test();
	//cv_gradient_test();
	//cv_draw_test();
	cv_kd_tree_test();

	std::cout << "libcv test passed!" << std::endl;

	return EXIT_SUCCESS;
}
