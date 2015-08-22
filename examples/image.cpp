
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
// Example program for usage of cv::image_array, gui and io, and of some procedures form improc.


#include <iostream>
#include <algorithm>

#include "../include/improc.hpp"
#include "../include/image.hpp"
#include "../include/io.hpp"
#include "../include/gui.hpp"


int main(int argc, char **argv) {

	if (argc != 2) {
		std::cout << "Invalid arguments - add path to an png or jpg image" << std::endl;
		return EXIT_FAILURE;
	}

	cv::image_array im = cv::imread(argv[1], cv::UINT8, 1);
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

	return EXIT_SUCCESS;
}

