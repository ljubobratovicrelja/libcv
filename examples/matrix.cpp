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


#include <iostream>
#include <algorithm>

#include "../include/vector.hpp"

/*
	matrix class check module
*/

void cv_vector_unittest() {
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
}

int main() {

	cv_vector_unittest();

	/*
	cv::matrixf nullmatrix;
	std::cout << "Null matrix:\n" << nullmatrix << std::endl;

	cv::matrixf sizedmatrix(3, 3);
	std::cout << "Sized matrix:\n" << sizedmatrix << std::endl;

	cv::matrixf parentmatrix(5, 5, 5);
	std::cout << "Parent matrix:\n" << parentmatrix << std::endl;

	cv::matrixf submatrix(1, 1, 3, 3, parentmatrix, RoiType::REFERENCE);
	submatrix.fill(3);
	std::cout << "Sub matrix:\n" << submatrix << std::endl;
	std::cout << "Parent matrix After Submatrix Change:\n" << parentmatrix << std::endl;

	cv::matrixf submatrixCopy(1, 1, 3, 3, parentmatrix);
	submatrixCopy.fill(7);
	std::cout << "Sub matrix Copy:\n" << submatrixCopy << std::endl;
	std::cout << "Parent matrix After Submatrix Change:\n" << parentmatrix << std::endl;

	cv::matrix3i imagematrix(3, 3, vec3i({15, 15, 15}));
	std::cout << "Image matrix:\n" << imagematrix << std::endl;

	std::cout << "Press any key to exit..." << std::endl;
	*/


	getchar();

	return EXIT_SUCCESS;
}
