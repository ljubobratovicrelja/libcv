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


#include "../include/improc.hpp"


namespace cv {

matrixr gauss(const vec2i &kernel_size, real_t theta) {
	ASSERT(kernel_size[0] > 2 && kernel_size[1] > 2);
	matrixr kernel(kernel_size[0], kernel_size[1]);

	unsigned midPoint_r = (kernel_size[0] / 2);
	unsigned midPoint_c = (kernel_size[1] / 2);

	for (unsigned i = 0; i < kernel_size[0]; ++i) {
		for (unsigned j = 0; j < kernel_size[1]; ++j) {
		kernel(i, j) = (1 / (2 * PI * pow(theta, 2)))
			* exp(-((pow(abs(midPoint_c - j), 2) + pow(abs(midPoint_r - i), 2)) / (2 * pow(theta, 2))));
		}
	}

	return kernel;
}

matrixr conv(const matrixr &in, const matrixr &conv_kernel) {
	ASSERT(in && conv_kernel);

	matrixr out(in.size());

	size_t convRows = conv_kernel.rows();
	size_t convCols = conv_kernel.cols();

	int convRows_half = convRows / 2;
	int convCols_half = convCols / 2;

	#pragma omp parallel
	{
		matrixr in_kernel(conv_kernel.rows(), conv_kernel.cols());

		real_t pixel_value;
		int conv_r, conv_c, rstart, rend, cstart, cend;
		int pixpos[2];

		#pragma omp for
		for (unsigned r = 0; r < in.rows(); ++r) {
			for (unsigned c = 0; c < in.cols(); ++c) {

				pixel_value = 0;
				conv_r = 0;
				rstart = r - convRows_half;
				rend = rstart + convRows;
				cstart = c - convCols_half;
				cend = cstart + convCols;

				for (int kr = rstart; kr < rend; ++kr) {
					pixpos[0] = ((kr < 0) ? abs(kr) : (kr >= in.rows() ? (in.rows() - 1) - (kr - in.rows()) : kr));
					conv_c = 0;
					for (int kc = cstart; kc < cend; ++kc) {
						pixpos[1] = ((kc < 0) ? abs(kc) : (kc >= in.cols() ? (in.cols() - 1) - (kc - in.cols()) : kc));
						pixel_value += in(pixpos[0], pixpos[1]) * conv_kernel(conv_r, conv_c);
						conv_c++;
					}
					conv_r++;
				}
				out(r, c) = pixel_value;
			}
		}
	}

	return out;
}

matrixr harris(const matrixb &in, unsigned windowvec2i, real_t kValue, real_t gaussTheta) {

	ASSERT(in);

	matrixf in_d = in;
	matrixf res = matrixf::zeros(in.size());
	real_t gaus_del = 2.f * pow(gaussTheta, 2);

	matrixf fx, fy;
	calc_derivatives(in_d, fx, fy);

#pragma omp parallel
	{
		real_t R, T;  // Score value
		real_t gauss_val = 1, _gx, _gy, _r1, _r2, _r3;

#pragma omp for schedule(dynamic)
		for (unsigned i = 0; i < in.rows(); i++)
			for (unsigned j = 0; j < in.cols(); j++)

			{
				if ((i + windowvec2i / 2 > in.rows() - 1) || (j + windowvec2i / 2 > in.cols() - 1)) {
					continue;
				}

				_r1 = _r2 = _r3 = .0f;

				for (unsigned cr = i - windowvec2i / 2; cr < i + windowvec2i / 2; cr++) {
					for (unsigned cc = j - windowvec2i / 2; cc < j + windowvec2i / 2; cc++) {

						gauss_val = exp(-(((i - cr) * (i - cr)) + ((j - cc) * (j - cc))) / gaus_del);

						_gx = fx(cr, cc);
						_gy = fy(cr, cc);
						_r1 += gauss_val * (_gx * _gx);
						_r2 += gauss_val * (_gx * _gy);
						_r3 += gauss_val * (_gy * _gy);
					}
				}

				T = _r1 + _r3;
				R = ((_r1 * _r3) - (_r2 * _r2)) - kValue * (T * T);

				if (R > 0)
					res(i, j) = R;
			}
	}
	return res;
}
}

