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


#include "../include/improc.hpp"


namespace cv {

namespace internal {

struct feature_par_cmp {
	bool operator () (const std::pair<vec2r, real_t> &r1, const std::pair<vec2r, real_t> &r2) const {
		return (r1.second < r2.second);
	}
};

}

matrixr gauss(const vec2i &kernel_size, real_t theta) {
	ASSERT(kernel_size[0] > 2 && kernel_size[1] > 2);
	matrixr kernel(kernel_size[0], kernel_size[1]);

	int midPoint_r = (kernel_size[0] / 2);
	int midPoint_c = (kernel_size[1] / 2);

	for(int i = 0; i < kernel_size[0]; ++i) {
		for(int j = 0; j < kernel_size[1]; ++j) {
			kernel(i, j) = (1 / (2 * PI * pow(theta, 2))) * exp(-((pow(abs(midPoint_c - j), 2) + pow(abs(midPoint_r - i), 2)) / (2 * pow(theta, 2))));
		}
	}

	cv::normalize(kernel.begin(), kernel.end(), Norm::L1);

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
		for (int r = 0; r < in.rows(); ++r) {
			for (int c = 0; c < in.cols(); ++c) {

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

matrixr threshold(const matrixr &in, real_t low_thresh, real_t up_thresh) {

	matrixr th = in.clone();

	for(unsigned i = 0; i < in.rows(); ++i) {
		for(unsigned j = 0; j < in.cols(); ++j) {
			if (in(i, j) >= low_thresh && in(i, j) <= up_thresh)
				th(i, j) = 1;
			else
				th(i, j) = 0;
		}
	}
	return th;
}

template<class corner_detector>
matrixr calc_corners(const matrixr &in, unsigned win_size, real_t gaussTheta, corner_detector cd) {

	ASSERT(in);

	matrixr res = matrixr::zeros(in.size());
	real_t gaus_del = 2.f * pow(gaussTheta, 2);

	matrixr fx, fy;
	calc_derivatives(in, fx, fy);

	auto win_sqr = win_size * win_size;

#pragma omp parallel
	{
		real_t R;  // Score value
		real_t gauss_val = 1, _gx, _gy, _r1, _r2, _r3;

#pragma omp for schedule(dynamic)
		for (int i = 0; i < in.rows(); i++)
			for (int j = 0; j < in.cols(); j++) {
				if ((i + win_size / 2 > in.rows() - 1) || (j + win_size / 2 > in.cols() - 1)) {
					continue;
				}
				_r1 = 0.;
				_r2 = 0.;
				_r3 = 0.;
				for (int cr = i - win_size / 2; cr < i + win_size / 2; cr++) {
					for (int cc = j - win_size / 2; cc < j + win_size / 2; cc++) {
						gauss_val = exp(-(((i - cr) * (i - cr)) + ((j - cc) * (j - cc))) / gaus_del);
						_gx = fx(cr, cc);
						_gy = fy(cr, cc);
						_r1 += gauss_val * (_gx * _gx);
						_r2 += gauss_val * (_gx * _gy);
						_r3 += gauss_val * (_gy * _gy);
					}
				}
				_r1 = (_r1 / win_sqr) * 0.5;
				_r2 /= win_sqr;
				_r3 = (_r3 / win_sqr) * 0.5;
				R = cd(_r1, _r2, _r3);
				if (R > 0) {
					res(i, j) = R;
				}
			}
	}
	return res;
}

struct harris_detector {
	real_t k;

	harris_detector(real_t k) : k(k) {}

	real_t operator()(real_t r1, real_t r2, real_t r3) {
		return (((r1 * r1) - (r2 * r3)) -  k*((r1+r3) * r1+r3));
	}
};

struct shi_tomasi_detector {
	real_t operator()(real_t r1, real_t r2, real_t r3) {
		return ((r1 + r3) - std::sqrt((r1 - r3) * (r1 - r3) + r2 * r2));
	}
};

matrixr harris(const matrixr &in, unsigned win_size, real_t k, real_t gauss) {
	return calc_corners(in, win_size, gauss, harris_detector(k)); 
}

matrixr good_features(const matrixr &in, unsigned  win_size, real_t gauss) {
	return calc_corners(in, win_size, gauss, shi_tomasi_detector());
}

std::vector<vec2r> extract_features(const matrixr &in, int count) {
	
	std::vector<std::pair<vec2r, real_t> > f_vals;

	for(unsigned i = 0; i < in.rows(); ++i) {
		for(unsigned j = 0; j < in.cols(); ++j) {
			auto v = in(i, j);
			if (v > 0.) {
				f_vals.push_back({{static_cast<real_t>(j), static_cast<real_t>(i)}, v});
			}
		}
	}

	if (f_vals.empty())
		return std::vector<vec2r>();
	
	std::sort(f_vals.begin(), f_vals.end(), internal::feature_par_cmp());

	if (count == -1 || static_cast<size_t>(count) >= f_vals.size())
		count = f_vals.size();

	std::vector<vec2r> features;
	features.reserve(count);
	for (int i = f_vals.size() - 1; i >= f_vals.size()- count; --i) {
		features.push_back(f_vals[i].first);
	}

	return features;
}

}

