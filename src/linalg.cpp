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
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#include "../include/linalg.hpp"
#include "../include/math.hpp"

#include <lapacke.h>

extern "C" {
#include <cblas.h>
}

namespace cv {

#ifdef CV_REAL_TYPE_DOUBLE
#define cv_getrf LAPACKE_dgetrf
#define cv_getri LAPACKE_dgetri
#define cv_gesvd LAPACKE_dgesvd
#define cv_gesv LAPACKE_dgesv
#define cv_geev LAPACKE_dgeev
#define cv_gemm cblas_dgemm

#else

#define cv_getrf LAPACKE_sgetrf
#define cv_getri LAPACKE_sgetri
#define cv_gesvd LAPACKE_sgesvd
#define cv_gesv LAPACKE_sgesv
#define cv_geev LAPACKE_sgeev
#define cv_gemm cblas_sgemm
#endif


void invert(matrixr &matrix) {

	ASSERT(matrix && matrix.is_square());

	int n = matrix.rows();
	int *piv = new int[n];
	int info;

	info = cv_getrf(LAPACK_ROW_MAJOR, n, n, matrix.data(), n, piv);

	if (info > 0) {
		std::cerr <<("Argument " + std::to_string(info) + ", has an illegal value.");
	} else if (info < 0) {
		std::cerr <<(
				"matrix(" + std::to_string(info) + ", " + std::to_string(info)
						+ ") is exactly zero. The factorization has " "been completed, but the factor U is exactly singular, and division " "by zero will occur if it is used to solve a system of equations.");
	}

	info = cv_getri(LAPACK_ROW_MAJOR, n, matrix.data(), n, piv);

	if (info > 0) {
		std::cerr <<("Argument " + std::to_string(info) + ", has an illegal value.");
	} else if (info < 0) {
		std::cerr <<(
				"matrix(" + std::to_string(info) + ", " + std::to_string(info)
						+ ") is exactly zero. The factorization has " "been completed, but the factor U is exactly singular, and division " "by zero will occur if it is used to solve a system of equations.");
	}

	delete[] piv;
}

size_t rank(const matrixr &matrix) {
	size_t r = 0;

	matrixr U, S, Vt;
	svd_decomp(matrix, U, S, Vt);

	float tol = std::max(matrix.rows(), matrix.cols()) * eps((real_t) * std::max_element(S.begin(), S.end()));

	for (unsigned i = 0; i < std::min(S.rows(), S.cols()); ++i) {
		if (S(i, i) > tol) {
			r++;
		}
	}
	return r;
}

void lu_decomp(const matrixr &A, matrixr &L, matrixr &U, matrixr &P) {

	ASSERT(A);

	int rows = A.rows();
	int cols = A.rows();
	int lda = std::max(rows, cols);
	int *piv = new int[rows];
	int info;

	matrixr work = A.clone();

	info = cv_getrf(LAPACK_ROW_MAJOR, rows, cols, work.data(), lda, piv);

	delete[] piv;

	if (info > 0) {
		std::cerr <<("Argument " + std::to_string(info) + ", has an illegal value.");
	} else if (info < 0) {
		std::cerr <<(
				"matrix(" + std::to_string(info) + ", " + std::to_string(info)
						+ ") is exactly zero. The factorization has " "been completed, but the factor"
								" U is exactly singular, and division " "by zero will occur if it is used "
								"to solve a system of equations.");
	}

	L = matrixr::zeros(work.size());
	U = matrixr::zeros(work.size());

	for(unsigned i = 0; i < work.rows(); ++i) {
		for(unsigned j = 0; j < work.cols(); ++j) {
			if (j >= i) {
				U(i, j) = work(i, j);
				if (i == j) {
					L(i, j) = 1;
				}
			} else {
				L(i, j) = work(i, j);
			}
		}
	}

	matrixr LU = L * U;
	invert(LU);
	P = A * LU;
	P.transpose();
}

matrixr svd_decomp(const matrixr &A, matrixr &U, matrixr &S, matrixr &VT) {

	ASSERT(A);

	int m = A.rows();
	int n = A.cols();
	int lda = n;
	int ldu = m;
	int ldvt = n;
	int d = std::min(m, n);
	int info;

	matrixr modifA(A);

	real_t *a = modifA.data();

	real_t *superb = (real_t*) malloc((std::min(m, n) - 1) * sizeof(real_t));
	real_t *u = (real_t*) malloc(ldu * m * sizeof(real_t));
	real_t *s = (real_t*) malloc(d * sizeof(real_t));

	U.create(m, n);
	S = matrixr::zeros(d, d);
	VT.create(ldvt, n);

	info = cv_gesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, a, lda, s, u, ldu, VT.data(), ldvt, superb);

	if (info < 0) {
		std::cerr <<("Invalid LAPACKE argument " + std::to_string(info * -1));
	} else if (info > 0) {
		std::cerr <<("SVD computing algorithm failed to converge.\n");
	} else {
		for(int i = 0; i < m; ++i) {
			for(int j = 0; j < n; ++j) {
				U(i, j) = u[j + ldu * i];
			}
		}

		for (int i = 0; i < d; ++i) {
			S(i, i) = s[i];
		}
	}

	free(superb);
	free(u);
	free(s);

	return modifA;
}

void lu_solve(const matrixr &A, const matrixr &B, matrixr &X) {

	ASSERT(A && B && A.is_square() && A.cols() == B.rows());

	matrixr aWork = A.clone();
	X = B.clone();

	int info;
	int n = A.rows();
	int lda = n;
	int *piv = new int[n];
	int nrhs = X.cols();
	int ldb = X.cols();

	info = cv_gesv(LAPACK_ROW_MAJOR, n, nrhs, aWork.data(), lda, piv,
			X.data(), ldb);

	if (info > 0) {
		printf("The diagonal element of the triangular factor of A,\n");
		printf("U(%i,%i) is zero, so that A is singular;\n", info, info);
		printf("the solution could not be computed.\n");
		X.release();
		return;
	}

	delete[] piv;
}

void null_solve(const matrixr &in, matrixr &nullSpace, bool normalize) {
	ASSERT(in);

	matrixr U, S, Vt;
	svd_decomp(in, U, S, Vt);

	if (!Vt) {
		std::cerr <<("Invalid SVD factorization.");
		return;
	}

	Vt.transpose();
	nullSpace.create(Vt.rows(), 1);

	for (unsigned i = 0; i < Vt.rows(); ++i) {
		nullSpace(i, 0) = Vt(i, Vt.cols() - 1);
	}

	if(normalize)
		nullSpace *= 1 / Vt(Vt.rows() - 1, Vt.cols() - 1);
}

void gemm(const matrixr &A, const matrixr &B, matrixr &C, real_t alpha, real_t beta) {

	ASSERT(A && B && A.cols() == B.rows());

	if (beta && (!C || !(C.rows() == A.rows() && C.cols() == B.cols()))) {
		C.create(A.rows(), B.cols());
	} else if (!beta) {
		C.create(A.rows(), B.cols());
	}

	int m = A.rows();
	int k = A.cols();
	int n = B.cols();

	cv_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, 
			A.data(), k, B.data(), n, beta, C.data(), n);
}

void eigenvalues(const matrixr &in, vector<real_t> &wr, vector<real_t> &wi) {
	ASSERT(in.is_square());
	wr.create(in.rows());
	wi.create(in.rows());
	matrixr work = in.clone();
	size_t size = work.rows();
	int info = cv_geev(LAPACK_ROW_MAJOR, 'N', 'N', size, work.data(),
			size, wr.data(), wi.data(), 0, size, 0, size);
	if (info) {
		std::cerr << "Error calculating eigenvalues.";
	}
}

void eigenvectors(const matrixr &in, vector<real_t> &lv, vector<real_t> &rv) {
	ASSERT(in.is_square());
	vector<real_t> wr(in.rows());
	vector<real_t> wi(in.rows());
	matrixr work = in.clone();
	lv.create(in.rows());
	rv.create(in.rows());
	size_t size = work.rows();
	int info = cv_geev(LAPACK_ROW_MAJOR, 'V', 'V', size, work.data(),
			size, wr.data(), wi.data(), lv.data(), size, rv.data(),
			size);
	if (info) {
		std::cerr << "Error calculating eigenvalues.";
	}
}

void geev(const matrixr &in, vector<real_t> &wr, vector<real_t> &wi, matrixr &lv, matrixr &rv) {
	ASSERT(in.is_square());
	wr.create(in.rows());
	wi.create(in.rows());
	matrixr work = in.clone();
	lv.create(in.size());
	rv.create(in.size());
	size_t size = work.rows();
	int info = cv_geev(LAPACK_ROW_MAJOR, 'V', 'V', size, work.data(),
			size, wr.data(), wi.data(), lv.data(), size, rv.data(),
			size);
	if (info) {
		std::cerr << "Error calculating eigenvalues.";
	}
}

int rodrigues_solve(const matrixr &src, matrixr &dst, matrixr *jacobian) {
	int depth, elem_size;
	int i, k;

	if (jacobian) {
		if ((jacobian->rows() != 9 || jacobian->cols() != 3)
				&& (jacobian->rows() != 3 || jacobian->cols() != 9)) {
			throw std::runtime_error("Jacobian must be 3x9 or 9x3");
		}
	}

// Conversion from vector to matrix.
	if (src.cols() == 1 || src.rows() == 1) {
		real_t rx, ry, rz, theta;
		int step = 0;

		if (src.rows() * src.cols() != 3)
			throw std::runtime_error("Input matrix must be 1x3, 3x1 or 3x3");

		rx = src.data_begin()[0];
		ry = src.data_begin()[1];
		rz = src.data_begin()[2];

		theta = sqrt(rx * rx + ry * ry + rz * rz);

		auto jacobian_data = jacobian->data_begin();
		if (theta < std::numeric_limits<real_t>::epsilon()) {
			dst.to_identity();
			if (jacobian) {
				jacobian->fill(0);
				jacobian_data[5] = jacobian_data[15] = jacobian_data[19] = -1;
				jacobian_data[7] = jacobian_data[11] = jacobian_data[21] = 1;
			}
		} else {
			real_t Imat[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

			real_t c = cos(theta);
			real_t s = sin(theta);
			real_t c1 = 1. - c;
			real_t itheta = theta ? 1. / theta : 0.;

			rx *= itheta;
			ry *= itheta;
			rz *= itheta;

			real_t rrt[] = { rx * rx, rx * ry, rx * rz, rx * ry, ry * ry, ry
					* rz, rx * rz, ry * rz, rz * rz };
			real_t _r_x_[] = { 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };
			matrixr matR(3, 3);

			for (k = 0; k < 9; k++)
				matR.data()[k] = c * Imat[k] + c1 * rrt[k] + s * _r_x_[k];

			dst = matR.clone();

			if (jacobian) {
				real_t drrt[] = { rx + rx, ry, rz, ry, 0, 0, rz, 0, 0, 0, rx, 0,
						rx, ry + ry, rz, 0, rz, 0, 0, 0, rx, 0, 0, ry, rx, ry,
						rz + rz };
				real_t d_r_x_[] = { 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, 0, 0,
						0, -1, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0 };
				for (i = 0; i < 3; i++) {
					real_t ri = i == 0 ? rx : i == 1 ? ry : rz;
					real_t a0 = -s * ri, a1 = (s - 2 * c1 * itheta) * ri, a2 =
							c1 * itheta;
					real_t a3 = (c - s * itheta) * ri, a4 = s * itheta;
					for (k = 0; k < 9; k++)
						jacobian->data_begin()[i * 9 + k] = a0 * Imat[k] + a1 * rrt[k]
								+ a2 * drrt[i * 9 + k] + a3 * _r_x_[k]
								+ a4 * d_r_x_[i * 9 + k];
				}
			}
		}
	} else if (src.cols() == 3 && src.rows() == 3) {
		real_t rx, ry, rz;
		matrixr R(3, 3);
		matrixr U(3, 3);
		matrixr V(3, 3);
		matrixr W(3, 1);
		real_t theta, s, c;

		dst.create(3, 1);

		R = src.clone();

		auto minMaxVals = std::minmax_element(R.begin(), R.end());

		if (*minMaxVals.first < -100 || *minMaxVals.second > 100) {
			dst.to_zero();
			if (jacobian)
				jacobian->to_zero();
			return 0;
		}

		R = svd_decomp(R, U, W, V);
		gemm(U, V, R);

		rx = R.data_begin()[7] - R.data_begin()[5];
		ry = R.data_begin()[2] - R.data_begin()[6];
		rz = R.data_begin()[3] - R.data_begin()[1];

		s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
		c = (R.data_begin()[0] + R.data_begin()[4] + R.data_begin()[8] - 1) * 0.5;
		c = c > 1. ? 1. : c < -1. ? -1. : c;
		theta = acos(c);

		if (s < 1e-5) {
			real_t t;

			if (c > 0)
				rx = ry = rz = 0;
			else {
				t = (R.data_begin()[0] + 1) * 0.5;
				rx = std::sqrt(std::max(t, static_cast<real_t>(0)));
				t = (R.data_begin()[4] + 1) * 0.5;
				ry = std::sqrt(std::max(t, static_cast<real_t>(0))) * (R.data_begin()[1] < 0 ? -1. : 1.);
				t = (R.data_begin()[8] + 1) * 0.5;
				rz = std::sqrt(std::max(t, static_cast<real_t>(0))) * (R.data_begin()[2] < 0 ? -1. : 1.);
				if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz)
						&& (R.data_begin()[5] > 0) != (ry * rz > 0))
					rz = -rz;
				theta /= sqrt(rx * rx + ry * ry + rz * rz);
				rx *= theta;
				ry *= theta;
				rz *= theta;
			}

			if (jacobian) {
				jacobian->to_zero();
				if (c > 0) {
					jacobian->data_begin()[5] = jacobian->data_begin()[1] = jacobian->data_begin()[19] = -0.5;
					jacobian->data_begin()[7] = jacobian->data_begin()[11] = jacobian->data_begin()[21] = 0.5;
				}
			}

		} else {
			real_t vth = 1 / (2 * s);

			if (jacobian) {
				real_t dtheta_dtr = -1. / s;

				real_t dvth_dtheta = -vth * c / s;
				real_t d1 = 0.5 * dvth_dtheta * dtheta_dtr;
				real_t d2 = 0.5 * dtheta_dtr;

				real_t dvardR[5 * 9] = { 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1,
						0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, d1, 0, 0,
						0, d1, 0, 0, 0, d1, d2, 0, 0, 0, d2, 0, 0, 0, d2 };

				real_t dvar2dvar[] = { vth, 0, 0, rx, 0, 0, vth, 0, ry, 0, 0, 0,
						vth, rz, 0, 0, 0, 0, 0, 1 };
				real_t domegadvar2[] = { theta, 0, 0, rx * vth, 0, theta, 0, ry
						* vth, 0, 0, theta, rz * vth };

				matrixr _dvardR(5, 9, dvardR);
				matrixr _dvar2dvar(4, 5, dvar2dvar);
				matrixr _domegadvar2(3, 4, domegadvar2);

				matrixr _t0(3, 5);

				gemm(_domegadvar2, _dvar2dvar, _t0);
				gemm(_t0, _dvardR, *jacobian);

				jacobian->transpose();
			}

			vth *= theta;
			rx *= vth;
			ry *= vth;
			rz *= vth;
		}

		dst.data_begin()[0] = rx;
		dst.data_begin()[1] = ry;
		dst.data_begin()[2] = rz;

	} else {
		throw std::runtime_error("Invalid input size - should be 3x1, 1x3 or 3x3.");
	}

	return 1;
}

void decompose_eigenvector_matrix(matrixr ev_mat, std::vector<vectorr> &evs, bool normalize) {
	ASSERT(ev_mat && ev_mat.is_square());
	evs.clear();
	for (unsigned i = 0; i < ev_mat.cols(); ++i) {
		evs.push_back(ev_mat.col(i));
		if (normalize) {
			evs.back() *= 1. / evs.back()[0];
		}
	}
}

vectorr get_eigenvector(matrixr ev_mat, unsigned eig_val_idx, bool normalize) {
	ASSERT(ev_mat && eig_val_idx < ev_mat.cols());
	auto ev = ev_mat.col(eig_val_idx);
	if (normalize)
		ev *= 1. / ev[0];
	return ev;
}
}
