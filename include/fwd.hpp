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

#define LOOP_FOR(from,to,by) for(int i=from;i<to;i+=by)
#define LOOP_FOR_TO(to) LOOP_FOR(0,to,1)
#define NEST_FOR(from,to,by,from_nest,to_nest,by_nest) for(int i=from;i<to;i+=by) for(int j=from_nest;j<to_nest;j+=by_nest)
#define NEST_FOR_TO(to,to_nest)  NEST_FOR(0,to,1,0,to_nest,1)

#define OMP_PARALLEL_FOR CV_PRAGMA(omp parallel for) //!< OpenMP parallel for pragma initializer macro.
#define OMP_INIT CV_PRAGMA(omp parallel) { //!< OpenMP parallel pragma initializer macro.
#define OMP_END } //!< OpenMP parallel for pragma deinitializer macro.
#define OMP_FOR CV_PRAGMA(omp for) //!< OpenMP for for pragma initializer macro.
#define OMP_ATOMIC CV_PRAGMA(omp atomic) //!< Perform OpenMP atomic increment.
#define OMP_CRITICAL CV_PRAGMA(omp critical) //!< Start OpenMP critical segment.
#define LOOP_OMP_FOR(cnti) OMP_FOR LOOP_FOR_TO(cnti) //!< OpenMP for pragma initializer with loop macro.
#define LOOP_PARALLEL_FOR(from,to,by) OMP_PARALLEL_FOR LOOP_FOR(from,to,by) //!< OpenMP parallel for pragma initializer with loop macro.
#define LOOP_PARALLEL_FOR_TO(cnt) OMP_PARALLEL_FOR LOOP_FOR_TO(cnt) //!< OpenMP parallel for pragma initializer macro.
#define NEST_PARALLEL_FOR_TO(cnti, cntj) LOOP_PARALLEL_FOR_TO(cnti) for(int j = 0; j < cntj; j++) //!< OpenMP parallel for pragma initializer with nested loop macro.
#define NEST_OMP_FOR_TO(cnti, cntj) OMP_FOR NEST_FOR_TO(cnti, cntj) //!< OpenMP parallel for pragma initializer with nested loop macro.
#define FOREACH(iter, cnt) for(auto iter = cnt.begin(); iter != cnt.end(); iter++)
#define ITERATE(iter, begin, end) for (auto iter = begin; iter != end; iter++)

typedef unsigned char byte;
typedef byte uchar;
typedef unsigned int uint;

#ifdef CV_REAL_TYPE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#define INTERPRET_INDEX(idx,length) (idx >= 0) ? idx : length + idx

enum parallelization_module {
	CV_OPENMP,
	CV_TBB,
	CV_OCL
};

#ifndef CV_PARALLELIZATION_MODULE
#ifdef _OPENMP
#define CV_PARALLELIZATION_MODULE CV_OPENMP
#endif

#endif

#ifndef VECTOR_PARALLELIZATION_THRESHOLD

/*!
	\def VECTOR_PARALLELIZATION_THRESHOLD

	This value represents a threshold after which some of algorithms over vectors will be paralellized on the CPU/GPU.
	For instance, small vectors (e.g. with size less than 1000) will take more time to allocate thread stack space
	(or allocating device memory for GPU), than to simply evaluate an algorithm on a single CPU core. For larger
	vectors (default value is 100000), parallelization of defined type (default is OpenMP) will be initialized.

	\note
	This value is initialized in fwd.hpp header, but will be overriden if preprocessor define with the same
	name is defined in compilation setup. (/DVECTOR_PARALLELIZATION_THRESHOLD=100000)
	*/
#define VECTOR_PARALLELIZATION_THRESHOLD 100000
#endif

#ifndef CV_BPQ_INIT_VALUE
#define CV_BPQ_INIT_VALUE 10e+30
#endif

typedef unsigned refcount_type;

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

enum ColorConvertion {

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
