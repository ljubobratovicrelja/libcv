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
// Simple array structure used specifically to note index arrays, mainly
// used in basic_array structure in array.hpp.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef INDEXARRAY_HPP_HWUG35OL
#define INDEXARRAY_HPP_HWUG35OL

#ifndef INDEX_ARRAY_SIZE
#define INDEX_ARRAY_SIZE 3
#endif


#include <iostream>
#include <initializer_list>

#include "fwd.hpp"


namespace cv {

/*!
 * @brief Simple array structure useful for storing indices for n-D arrays.
 */
class CV_EXPORT index_array {
private:
	unsigned _data[INDEX_ARRAY_SIZE]; // data of the array.
	unsigned _size; // used data in the array.
public:
	//! Default constructor.
	index_array();
	//! Constructor using size.
	index_array(unsigned size);
	//! Constructor with initializer list.
	index_array(const std::initializer_list<unsigned> &list);

	//! Get size of the array.
	unsigned size() const;
	//! Check if size is zero.
	bool empty() const;
	//! Index the array.
	unsigned &operator[](unsigned i);
	//! Read-only indexing of the array.
	unsigned operator[](unsigned i) const;

	//! Get beginning pointer to the array.
	unsigned *begin();
	//! Get the ending pointer to the array.
	unsigned *end();
	//! Get read-only pointer to the array.
	const unsigned *begin() const;
	//! Get read-only pointer to the end of the array.
	const unsigned *end() const;
	//! Get sum of the array values. 
	unsigned sum() const;
	//! Get product of each array value. Useful as count of the n-d array size.
	unsigned product() const;
	//! Resize the array. Should be resized to the number smaller than INDEX_ARRAY_SIZE value. Else throws.
	void resize(unsigned size);
	//! Clears the array.
	void clear();

	bool operator==(const index_array &rhs) const;
	bool operator!=(const index_array &rhs) const;
	bool operator<(const index_array &rhs) const;

	index_array operator+(const index_array& rhs) const;
	index_array operator-(const index_array& rhs) const;
	index_array operator/(const index_array& rhs) const;
	index_array operator*(const index_array& rhs) const;

	index_array &operator+=(const index_array& rhs);
	index_array &operator-=(const index_array& rhs);
	index_array &operator/=(const index_array& rhs);
	index_array &operator*=(const index_array& rhs);

	index_array operator+(unsigned rhs) const;
	index_array operator-(unsigned rhs) const;
	index_array operator/(unsigned rhs) const;
	index_array operator*(unsigned rhs) const;

	index_array &operator+=(unsigned rhs);
	index_array &operator-=(unsigned rhs);
	index_array &operator/=(unsigned rhs);
	index_array &operator*=(unsigned rhs);

	friend std::ostream &operator << (std::ostream &stream, const index_array &array) {
		if (!array.empty()) {
			for (auto v : array) {
				stream << v << " ";
			}
		}
		return stream;
	}
};

}

#endif /* end of include guard: INDEXARRAY_HPP_HWUG35OL */
