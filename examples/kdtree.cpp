

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
// Example program for usage of cv::kd_tree.


#include <iostream>
#include <algorithm>

#include "../include/gui.hpp"
#include "../include/matrix.hpp"
#include "../include/kdtree.hpp"
#include "../include/draw.hpp"


int main() {

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
			cv::draw_circle(img, source[i], 5, cv::vec3b {0, 0, 255});
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
			return EXIT_SUCCESS;
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

	return EXIT_SUCCESS;
}

