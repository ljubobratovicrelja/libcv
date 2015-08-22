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
// Example program for usage of cv::vector and cv::vectorx


#include <iostream>
#include <algorithm>

#include "../include/vector.hpp"


int main() {

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

	return EXIT_SUCCESS;
}

