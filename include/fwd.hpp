//The MIT License (MIT)
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
// Core project header, containing build option preprocessor switches, and
// main macros used trough the library code. 
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef FWD_HPP_1F9LO7BS
#define FWD_HPP_1F9LO7BS


#if defined(_MSC_VER)
//  Microsoft Visual Compiler, and DLL compilation
#define CV_EXPORT __declspec(dllexport)
#define CV_IMPORT __declspec(dllimport)
#define CV_PRAGMA(arg) __pragma(arg)
#define CV_TYPENAME typename
#define CV_WINDOWS
#elif defined(__GNUG__) //  GNU Compiler
#define CV_EXPORT __attribute__((visibility("default")))
#define CV_PRAGMA(arg) _Pragma(#arg)
#define CV_IMPORT
#define CV_TYPENAME typename
#define CV_LINUX
#else // arbitrary compiler, not expected and handled.
#error Unrecognized compiler
#define CV_EXPORT
#define CV_IMPORT
#define CV_TYPENAME typename
#define CV_PRAGMA(arg) 
#endif

#define PI (double)3.14159265358979323846264338327950288  /*!< Pi value. */
#define RAD_TO_DEG(val) (double)( val * (180.0 / PI)) /*!< Convert radians to degrees. */
#define DEG_TO_RAD(val) (double)( val * (PI / 180.0)) /*!< Convert degrees to radians. */

typedef unsigned char byte;
typedef byte uchar;
typedef unsigned int uint;
typedef unsigned refcount_type;

#ifdef CV_REAL_TYPE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#define INTERPRET_INDEX(idx,length) (idx >= 0) ? idx : length + idx

#ifndef VECTOR_PARALLELIZATION_THRESHOLD
/*!
	\def VECTOR_PARALLELIZATION_THRESHOLD

	This value represents a threshold after which some of algorithms over vectors will be paralellized.
*/
#define VECTOR_PARALLELIZATION_THRESHOLD (unsigned long)1e+6
#endif

#ifndef CV_BPQ_INIT_VALUE
#define CV_BPQ_INIT_VALUE 10e+30
#endif

#define REF_INCREMENT(ref) if(ref) ++(*ref) //!< Increment reference counter - assumes given value (int*) is initialized.
#define REF_DECREMENT(ref) if(ref) --(*ref) //!< Decrement reference counter - assumes given value (int*) is initialized.
#define REF_CHECK(ref) ((*ref) <= 0) ? false : true //!< Check if reference counter is equal to 0 - assumes given value (int*) is initialized.
#define REF_NEW new refcount_type(1) //!< Instatiate new reference counter.
#define REF_DESTROY(ref) delete ref //!< Destroy reference counter.
#define REF_INIT(ref) REF_DESTROY(ref), ref = REF_NEW //!< Initialize reference counter, will first delete existing, then instantiate a new one.

/*!
 * Runtime assertion check.
 */
#if(defined(NDEBUG) || defined(NO_ASSERT))
#define ASSERT(cond)  // empty macro - skip runtime assertion checks in the release mode.
#else
#define ASSERT(cond) \
    do \
			    { \
        if (!(cond)) \
						        { \
            throw std::runtime_error(("Assertion check failed at condition: " #cond ", at line: ") + \
					std::to_string(__LINE__) + (", in file " __FILE__)); \
						        } \
				    } while(0)
#endif

/*!
	@brief Computer Vision core namespace.

	Contains data structures and basic algorithms used in
	image processing and computer vision tasks.

	Used as libcv base namespace.
	*/
namespace cv {

//! Type of region of interest data.
enum class RoiType {
	REFERENCE,  //!< Referenced data.
	COPY //!< Copy data.
};

//! Type of region of interest edge menagement.
enum class RoiEdge {
	MIRROR, //!< Mirrored edges.
	ZEROS //!< Zero edges.
};

//! Norm type.
enum class Norm {
	MINMAX,  //!< Min-Max norm.
	INF,  //!< Infinite norm.
	L0,  //!< Zero norm - L0.
	L1,  //!< L1 Norm.
	L1_SQRT,  //!< L! squared norm.
	L2,  //!< Eucledian norm.
};

enum class InterpolationType  //! Interpolation value_type used by some functions.
{
	NN, /*!< Nearest neighbour interpolation. */
	LINEAR, /*!< Bilinear interpolation. */
	CUBIC, /*!< Bicubic interpolation. */
	NO_INTER  //!< No interpolation used.
};

enum class ImageDepth {
	ANY = 0,
	BYTE_8 = 1,
	SHORT_16 = 2,
	FLOAT_32 = 4, 
	DOUBLE_64 = 8
};

enum class ImageFormat {
	ANY = 0,
	GRAY = 1,
	RGB = 3,
	RGBA = 4
};

namespace internal {
// Index comparator used for comparing arrays by only one axis values.
struct idx_cmp {
	unsigned index = 0;

	idx_cmp(unsigned index) : index(index) {}

	template<typename _ArrayType>
	bool operator ()(const _ArrayType &rhs, const _ArrayType &lhs) const {
		return (rhs[index] < lhs[index]);
	}
};

}
}

#endif /* end of include guard: FWD_HPP_1F9LO7BS*/
