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
// Collection of image processing algorithms.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef IMPROC_HPP_G3NILB69
#define IMPROC_HPP_G3NILB69


#include "fwd.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "rangefunc.hpp"
#include "matfunc.hpp"


namespace cv {

// common convolution kernel operator matrices

/*!
 * @brief Generate Gauss kernel operator
 */
matrixr CV_EXPORT gauss(const vec2i &kernelvec2i, real_t theta = .84);

/*!
 * @brief Image (2D) spatial convolution algorithm.
 */
matrixr CV_EXPORT conv(const matrixr &in, const matrixr &conv_kernel);

/*!
 * @brief Harris corner detector.
 */
matrixr CV_EXPORT harris(const matrixb &in, unsigned windowvec2i, real_t kValue, real_t gaussTheta);

template<typename _Tp, size_t cnt> inline
matrix<_Tp> color_to_gray(matrix<vectorx<_Tp, cnt> > in) {
	matrix<_Tp> out;

	if (!in)
		return out;

	out.create(in.size());

	switch (cnt) {
		case 1:
			NEST_FOR_TO(in.rows(), in.cols()) {
				out(i, j) = in(i, j)[0];
			}
			break;
		case 2:
			NEST_FOR_TO(in.rows(), in.cols()) {
				out(i, j) = in(i, j)[0];
			}
			break;
		case 3:
			NEST_FOR_TO(in.rows(), in.cols()) {
				out(i, j) = ranged_cast <_Tp> (in(i, j).mean());
			}
			break;
		case 4:
			NEST_FOR_TO(in.rows(), in.cols()) {
				real_t meanVal = 0;
				for (int c = 0; c < 3; c++) {
					meanVal += in(i, j)[c];
				}
				out(i, j) = _Tp(meanVal / 3);
			}
			break;
	}
	return out;
}

template <typename _Tp>
void calc_derivatives(const matrix<_Tp> &in, matrix<_Tp> &fx, matrix<_Tp> &fy) {

	ASSERT(in);

	fx.create(in.size());
	fy.create(in.size());

	// calc mid-ground
	for (unsigned r = 1; r < in.rows(); r++)
		for (unsigned c = 1; c < in.cols(); c++) {
			fx(r, c) = -in(r, c - 1) + in(r, c);
			fy(r, c) = -in(r - 1, c) + in(r, c);
		}
	// calc first edges
	for (unsigned r = 0; r < 1; r++)
		for (unsigned c = 0; c < 1; c++) {
			fx(r, c) = -in(r, c) + in(r, c + 1);
			fy(r, c) = -in(r, c) + in(r + 1, c);
		}
}

template<typename _Tp>
void filter_non_maximum(matrix<_Tp> &in, size_t filtervec2i) {

	ASSERT(in && filtervec2i);

	matrix<_Tp> lmsw;  // local maxima search window
	int lms_r, lms_c;
	int win_rows, win_cols;
	real_t lms_val;

	for (int br = 0; br < in.rows(); br += filtervec2i / 2) {
		for (int bc = 0; bc < in.cols(); bc += filtervec2i / 2) {
			win_rows = (br + filtervec2i < in.rows()) ? filtervec2i : filtervec2i - ((br + filtervec2i) - in.rows()) - 1;
			win_cols = (bc + filtervec2i < in.cols()) ? filtervec2i : filtervec2i - ((bc + filtervec2i) - in.cols()) - 1;

			if (win_rows <= 0 || win_cols <= 0) {
				continue;
			}

			lmsw.create(br, bc, win_rows, win_cols, in, RoiType::REFERENCE);

			lms_val = -1;
			for (int r = 0; r < lmsw.rows(); r++) {
				for (int c = 0; c < lmsw.cols(); c++) {
					if (lmsw(r, c) > lms_val) {
						lms_val = lmsw(r, c);
						lms_r = r;
						lms_c = c;
					}
				}
			}
			lmsw.toZero();
			if (lms_val != -1) {
				lmsw(lms_r, lms_c) = lms_val;
			}
		}
	}
}

template<typename _Tp>
real_t norm(const matrix<_Tp> &in, Norm ntype = Norm::L2) {
	ASSERT(in);

	real_t n = 0;

	switch(ntype) {
		case Norm::INF:
			n = std::numeric_limits<real_t>::min();
			for (unsigned i = 0; i < in.rows(); ++i) {
				for (unsigned j = 0; j < in.cols(); ++j) {
					if (in(i, j) > n) {
						n = in(i, j);
					}
				}
			}
			break;
		case Norm::L1:
			for (unsigned i = 0; i < in.rows(); ++i) {
				for (unsigned j = 0; j < in.cols(); ++j) {
					n += std::fabs(in(i, j));
				}
			}
			break;
		case Norm::L2:
			for (unsigned i = 0; i < in.rows(); ++i) {
				for (unsigned j = 0; j < in.cols(); ++j) {
					n += std::pow(in(i, j), 2);
				}
			}
			n = std::sqrt(n);
			break;
		default:
			throw std::runtime_error("Unsupported norm type - should be L1 - or L2");
	}

	return n;
}

template<typename _Tp>
matrix<_Tp> normalize(const matrix<_Tp> &in, Norm ntype = Norm::MINMAX, real_t lowv = 0., real_t hiv = 255.) {
	ASSERT(in);

	matrix<_Tp> out(in.size());

	if (ntype == Norm::MINMAX) {
		auto min = std::numeric_limits<real_t>::max();
		auto max = std::numeric_limits<real_t>::min();
		for (unsigned i = 0; i < in.rows(); ++i) {
			for (unsigned j = 0; j < in.cols(); ++j) {
				auto v = in(i, j);
				if (v > max)
					max = v;
				if (v < min)
					min = v;
			}
		}
		real_t sc_val = ((hiv - lowv) / (max - min)) + lowv;
		for (unsigned i = 0; i < in.rows(); ++i) {
			for (unsigned j = 0; j < in.cols(); ++j) {
				out(i, j) = (in(i, j) - min) * sc_val;
			}
		}
	} else if (ntype == Norm::L2 || ntype == Norm::L1 || ntype == Norm::INF) {

		auto normVal = norm(in, ntype);

		for (unsigned i = 0; i < in.rows(); ++i) {
			for (unsigned j = 0; j < in.cols(); ++j) {
				out(i, j) = in(i, j) / normVal;
			}
		}
	} else {
		throw std::runtime_error("Norm type not supported.");
	}
	return out;
}

}

#endif /* end of include guard: IMPROC_HPP_G3NILB69 */
