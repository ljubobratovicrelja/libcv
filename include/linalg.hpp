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
// Collection of basic linear algebra problem solving algorithms.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef LINALG_HPP_YW7IEDT1
#define LINALG_HPP_YW7IEDT1


#include "fwd.hpp"
#include "matrix.hpp"
#include "matfunc.hpp"


#ifndef CV_AUTO_NORM_EIGENVECTORS
#define CV_AUTO_NORM_EIGENVECTORS 0
#else
#if (CV_AUTO_NORM_EIGENVECTORS != 1 && CV_AUTO_NORM_EIGENVECTORS != 0)
#error "CV_AUTO_NORM_EIGENVECTORS has to be 1 or 0"
#endif
#endif

namespace cv {
/*!
 * @brief Compute the inverse of a matrix using the LU	factorization.
 *
 * This method inverts U and then computes inv(A) by solving the system
 * inv(A)*L = inv(U) for inv(A).
 */
void CV_EXPORT invert(matrixr &matrix);

/*!
 * @brief Find rank of the matrix.
 */
size_t CV_EXPORT rank(const matrixr &matrix);

/*!
 * @brief Compute an LU factorization of a general M-by-N matrix A using partial pivoting with	row interchanges.
 *
 * Computes an LU	factorization of a general M-by-N matrix
 * A using partial pivoting	with row interchanges. The factorization
 * has the form :
 * @code
 A = P * L * U
 @endcode
 * where P is a permutation matrix, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n), and U
 * is upper	triangular (upper trapezoidal if m < n).
 */
void CV_EXPORT lu_decomp(const matrixr &A, matrixr &L, matrixr &U,
		matrixr &P);

/*!
 * @brief Compute singular	value decomposition (SVD) of a real M-by-N
 * matrix A.
 *
 * @returns Factorized matrix A.
 */
matrixr CV_EXPORT svd_decomp(const matrixr &A, matrixr &U, matrixr &S,
		matrixr &VT);

/*!
 * @brief Solve a system of linear equations using LU decomposition.
 *
 * Solve a system of linear equations  A * X = B or A' * X = B
 * with a general N-by-N matrix	A using	the LU factorization.
 * A * X = B or A' * X = B with a general N-by-N matrix
 * A using	the LU factorization.
 *
 * @note
 * if matrix A is not square, system will perform:
 * At*A*X = At*B
 *
 * Uses lapack dgesv/sgesv
 *
 * \throws std::runtime_error
 */
void CV_EXPORT lu_solve(const matrixr &A, const matrixr &B, matrixr &X);

/*!
 * @brief Solve null space equation of a matrix.
 *
 * @param in Input matrix of arbitrary size.
 * @param nullSpace Null-space of the input matrix.
 */
void CV_EXPORT null_solve(const matrixr &in, matrixr &nullSpace, bool normalize = CV_AUTO_NORM_EIGENVECTORS);

/*!
 * @brief Perform general matrix multiplication.
 *
 * Wraps GEMM routine from BLAS: C = alpha*A*B + beta*C
 *
 * @param A Matrix of size mxk
 * @param B Matrix of size kxn
 * @param C If beta is 0, null matrix can be passed, in opposite, needs to be mxn. On exit is mxn.
 * @param alpha Scalar multiplication for matrix A.
 * @param beta Scalar multiplication for matrix C - pass 0 for operation C = AxB
 */
void CV_EXPORT gemm(const matrixr &A, const matrixr &B, matrixr &C,
		real_t alpha = 1.0, real_t beta = 0.0);

/*!
 * @brief Calculate eigenvalues of a square matrix.
 *
 * @param in Input matrix.
 * @param wr Vector containing real parts of eigenvalues.
 * @param wi Vector containing imaginary parts of eigenvalues.
 *
 * \throws std::runtime_error if input matrix is not a square matrix.
 */
void CV_EXPORT eigenvalues(const matrixr &in, vector<real_t> &wr, vector<real_t> &wi);

/*!
 * @brief Calculate eigenvectors of a square matrix.
 *
 * @param in Input matrix.
 * @param lv Vector containing left eigenvectors.
 * @param rv Vector containing right eigenvectors.
 *
 * \throws std::runtime_error if input matrix is not a square matrix.
 */
void CV_EXPORT eigenvectors(const matrixr &in, vector<real_t> &lv, vector<real_t> &rv);

/*!
 * @brief Calculate eigenvalues and eigenvectors of a square matrix. Wraps GEEV routine from Lapack.
 *
 * @param in Input matrix.
 * @param wr Vector containing real parts of eigenvalues.
 * @param wi Vector containing imaginary parts of eigenvalues.
 * @param lv Vector containing left eigenvectors.
 * @param rv Vector containing right eigenvectors.
 *
 * \throws std::runtime_error if input matrix is not a square matrix.
 */
void CV_EXPORT geev(const matrixr &in, vector<real_t> &wr, vector<real_t> &wi, matrixr &lv, matrixr &rv);

/*!
 * Evaluate Rodrigues rotation formula to convert rotation vector to matrix and vice-versa.
 *
 * @param src Input matrix, should be 3x1, 1x3 as vector or 3x3 as matrix.
 * @param dst Output matrix, if input is a vector output is a matrix and vice-versa.
 * @param jacobian If input, returns Jacobian matrix, which is a matrix of partial
 * derivatives of the output array components with respect to the input array components.
 *
 * @warning
 * Disclamer - implementation borrowed from opencv/calib3d module. (cvRodrigues2)
 * 
 * \note Jacobian matrix needs to be initialized to 9x3 or 3x9 if output is expected. If jacobian is
 * initialized differently, mta::Exception will be thrown.
 *
 * \throws std::runtime_error.
 *
 */
int CV_EXPORT rodrigues_solve(const matrixr &src, matrixr &dst, matrixr *jacobian);

/*!
 * @brief Decomposes eigenvector matrix from geev to individual vectors 
 * associated to each eigenvalue.
 *
 * @warning
 * Each eigen vector references matrix data.
 *
 * @param ev_mat eigenvalue matrix achieved from geev.
 * @param evs std::vector of cv::vectorr where individual vector columns will be stored.
 * @param normalize if true, vectors will be normalized by the first member to be 1.0.
 */
void CV_EXPORT decompose_eigenvector_matrix(matrixr ev_mat, std::vector<vectorr> &evs, bool normalize = CV_AUTO_NORM_EIGENVECTORS);

/*!
 * @brief Get eigenvalue associated with indexed eigenvalue.
 *
 * @param ev_mat eigenvalue matrix achieved from geev.
 * @param eig_val_idx index of the eigenvalue - column of the ev_mat.
 * @param normalize if true, vectors will be normalized by the first member to be 1.0.
 *
 * @throws if eig_val_idx is >= ev_mat.cols(), throws std::runtime_error.
 */
vectorr CV_EXPORT get_eigenvector(matrixr ev_mat, unsigned eig_val_idx, bool normalize = CV_AUTO_NORM_EIGENVECTORS);

}
#endif /* end of include guard: LINALG_HPP_YW7IEDT1 */
