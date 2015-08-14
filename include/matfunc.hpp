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
// Collection of matrix operations and functions.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef MATFUNC_HPP_PV3OUXCZ
#define MATFUNC_HPP_PV3OUXCZ


#include "array.hpp"
#include "matrix.hpp"


namespace cv {

template<typename _Tp> inline
void cross(const matrix<_Tp>& m1, const matrix<_Tp>& m2, matrix<_Tp>& out) {

	ASSERT(m1.rows() != 0 && m1.cols() != 0 && m1.cols() == m2.rows());

	if (!(out.rows() == m1.rows() && out.cols() == m2.cols())) {
		out.create(m1.rows(), m2.cols());
	}

	out.fill(0);

	NEST_FOR_TO(m1.rows(), m2.cols()) {
		for (int c = 0; c < m2.rows(); c++) {
			out(i, j) += m1(i, c) * m2(c, j);
		}
	}
}

/**
*  Transpose matrix operation.
*/
template<typename _Tp> inline
void transpose(const matrix<_Tp> &in, matrix<_Tp>& out) {
	ASSERT(in.rows() != 0 && in.cols() != 0);

	matrix<_Tp> workmatrix;
	if (&in == &out) {
		workmatrix.create(in);
	} else {
		workmatrix = in;
	}

	if (out.rows() != in.cols() || out.cols() != in.rows())
		out.create(in.cols(), in.rows());

	NEST_FOR_TO(workmatrix.rows(), workmatrix.cols()) {
		out(j, i) = workmatrix(i, j);
	}

}
/**
*  Calculate matrix determinant value.
*/
template<typename _Tp> inline
double determinant(const matrix<_Tp>& in) {

	ASSERT(in.rows() == in.cols());

	double det = 0.0;

	matrix<_Tp> minorImage(in);

	if (minorImage.rows() >= 3) {

		for (unsigned int col = 1; col < in.cols() + 1; col++) {

			minor_matrix(in, minorImage, 0, col - 1);
			det += pow(-1, (double)(col)) * in(0, col - 1) * determinant(minorImage);

		}

	} else if (minorImage.rows() == 2) {
		det = minorImage(0, 0) * minorImage(1, 1) - minorImage(1, 0) * minorImage(0, 1);
	} else if (minorImage.rows() == 1) {
		det = minorImage(0, 0);
	}

	return det;
}

/** @brief Return minor at member.
*
*  Returns minor matrix at given row and column index.
*/
template<typename _Tp> inline
void minor_matrix(const matrix<_Tp>& in, matrix<_Tp>& minormatrix, unsigned int row, unsigned int col) {

	ASSERT(row < in.rows() && col < in.cols());

	minormatrix.create(in.rows() - 1, in.cols() - 1);

	int detR = 0;
	int detC = 0;

	for (unsigned i = 0; i < in.rows(); i++) {
		for (unsigned j = 0; j < in.cols(); j++) {
			if (i != row && j != col) {
				minormatrix(detR, detC) = in(i, j);
				detC++;
			}
		}

		if (i != row) {
			detR++;
		}

		detC = 0;
	}
}

/**
*  matrix adjugation operation.
*/
template<typename _Tp> inline
void adjugate(const matrix<_Tp>& in, matrix<_Tp>& adjmatrix) {

	adjmatrix.create(in.rows(), in.cols());

	matrix<_Tp> min;
	int f;

	if (in.rows() == 2 && in.cols() == 2) {

		adjmatrix(0, 0) = in(1, 1);
		adjmatrix(0, 1) = in(0, 1) * -1;
		adjmatrix(1, 0) = in(1, 0) * -1;
		adjmatrix(1, 1) = in(0, 0);

	} else if (in.rows() > 2 && in.cols() > 2) {

		for (unsigned int r = 0; r < in.rows(); r++) {
			for (unsigned int c = 0; c < in.cols(); c++) {

				f = pow(-1, (double)(r + c));
				minor_matrix(in, min, r, c);
				adjmatrix(r, c) = f * determinant(min);

			}
		}

	}

	// transpose it at the end
	transpose(adjmatrix, adjmatrix);

}

/** @brief Resize matrix.
*
*  This method resizes this matrix object. As resize interpolation value_type current supported types of
*  interpolations present are nearest neighbour and INTER_LINEAR. Soon to be supported INTER_CUBIC.
*
*  @param[in] newRows New row count.
*  @param[in] newCols New column count.
*  @param[in] Interpolation value_type: enum INTERPOLATION_TYPE {INTERP_BILINEAR = 0, INTERP_BICUBIC, INTERP_NN};
*
*  @return
*  Reference to the resized matrix.
*/

template<typename _Tp>
void resize(matrix<_Tp> in, matrix<_Tp> &out, size_t newRows, size_t newCols,
	InterpolationType interp = InterpolationType::LINEAR) {


	// create out data matrix
	out.create(newRows, newCols);

	// prepare data
	float row_ratio = float(newRows - 1) / (in.rows() - 1);
	float col_ratio = float(newCols - 1) / (in.cols() - 1);

	// choose interpolation, then resize
	if (interp == InterpolationType::NN) {
		/*
		nearest neighbour interpolation - choose closest member
		*/
		OMP_PARALLEL_FOR
			for (unsigned r = 0; r < newRows; r++) {

				float r_old, c_old;
				int r_nn, c_nn;

				for (unsigned c = 0; c < newCols; c++) {

					r_old = (float)r / row_ratio;
					c_old = (float)c / col_ratio;

					r_nn = int(r_old);
					c_nn = int(c_old);

					if (r_nn >= in.rows())
						r_nn = in.rows() - 1;
					if (c_nn >= in.cols())
						c_nn = in.cols() - 1;

					out(r, c) = in(r_nn, c_nn);
				}
			}
	}

	if (interp == InterpolationType::LINEAR) {
		/*
		INTER_LINEAR interpolation
		f(x,y) = f(0,0)(1-x)(1-y) + f(1,0)x(1-y) + f(0,1)(1-x)y + f(1,1)xy;
		*/

		OMP_PARALLEL_FOR
			for (unsigned r = 0; r < newRows; r++) {

				_Tp c00, c01, c10, c11;

				float r_in, c_in;
				float r_diff, c_diff;

				int r_in_int, c_in_int;

				for (unsigned c = 0; c < newCols; c++) {

					r_in = (float)r / row_ratio;
					c_in = (float)c / col_ratio;

					r_in_int = (int)(r_in);
					c_in_int = (int)(c_in);

					r_diff = r_in - r_in_int;
					c_diff = c_in - c_in_int;

					c00 = in(r_in_int, c_in_int);
					c10 = (r_in_int + 1 < in.rows()) ?
						in(r_in_int + 1, c_in_int) : in(r_in_int, c_in_int);
					c01 = (c_in_int + 1 < in.cols()) ?
						in(r_in_int, c_in_int + 1) : in(r_in_int, c_in_int);

					if (r_in_int + 1 < in.rows() && c_in_int + 1 < in.cols()) {
						c11 = in(r_in_int + 1, c_in_int + 1);
					} else if (!(r_in_int + 1 < in.rows()) && c_in_int + 1 < in.cols()) {
						c11 = in(r_in_int, c_in_int + 1);
					} else if (r_in_int + 1 < in.rows() && !(c_in_int + 1 < in.cols())) {
						c11 = in(r_in_int + 1, c_in_int);
					} else {
						c11 = in(r_in_int, c_in_int);
					}

					float w00 = (1. - r_diff) * (1. - c_diff);
					float w01 = (1. - r_diff) * (c_diff);
					float w10 = (r_diff)* (1. - c_diff);
					float w11 = (r_diff)* (c_diff);

					_Tp res = _Tp(w00 * c00 + w01 * c01 + w10 * c10 + w11 * c11);

					out(r, c) = res;
				}
			}
	}
}

template<typename _Tp>
void resize(matrix<_Tp> in, matrix<_Tp> &out, vec2i newvec2i,
	InterpolationType interp) {
	resize(in, out, newvec2i[0], newvec2i[1], interp);
}

template<typename _Tp, typename _Tc>
_Tp ranged_cast(_Tc value) {

	double value_d = static_cast<double>(value);

	if (value_d < std::numeric_limits < _Tp > ::min())
		value_d = std::numeric_limits < _Tp > ::min();
	if (value_d > std::numeric_limits < _Tp > ::max())
		value_d = std::numeric_limits < _Tp > ::max();

	return static_cast<_Tp>(value_d);
}

template<typename _Tp>
_Tp trace(const matrix<_Tp>& in) {
	ASSERT(in.is_square());
	_Tp retVal = _Tp(0);
	LOOP_FOR_TO(in.rows()) {
		retVal += in(i, i);
	}
	return retVal;
}

template<typename _Tp>
_Tp sum(const matrix<_Tp> &in) {
	_Tp val = _Tp(0);
	for (unsigned i = 0; i < in.rows(); ++i) {
		for (unsigned j = 0; j < in.cols(); ++j) {
			val += in(i, j);
		}
	}
	return val;
}

template<typename _Tp>
double sum(const matrix<_Tp> &in, bool absolute) {
	double val = 0;

	if (absolute) {
		for (unsigned i = 0; i < in.rows(); ++i) {
			for (unsigned j = 0; j < in.cols(); ++j) {
				val += static_cast<double>(in(i));
			}
		}
	} else {
		for (unsigned i = 0; i < in.rows(); ++i) {
			for (unsigned j = 0; j < in.cols(); ++j) {
				val += fabs(in(i, j));
			}
		}
	}
	return val;
}

template<typename _Tp>
_Tp mean(const matrix<_Tp> &in) {
	return sum(in) / (in.rows() * in.cols());
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

template<typename _Tp>
real_t distance(const matrix<_Tp> &m1, const matrix<_Tp> &m2, Norm ntype = Norm::L2) {
	ASSERT(m1.size() == m2.size());
	return norm(m1 - m2, ntype);
}

}

#endif /* end of include guard: MATFUNC_HPP_PV3OUXCZ */
