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
// Bounded priority queue structure implementation.
// 
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef BPG_HPP_2QLEMDVV
#define BPG_HPP_2QLEMDVV


#include "fwd.hpp"
#include "vector.hpp"

#include <cstring>
#include <algorithm>


namespace cv {


/*!
 *
 * @brief Bounded priority Queue.
 *
 * Used to define limit sized queue,
 * arranged by priority score.
 *
 * As templates, queue takes a value type, and
 * comparator functor model, which is by default
 * std::less<real_t>.
 *
 * \warning
 * Consider the initial score value (priority_queue::set_init_score)
 * and comparator model together. Enqueue method performs
 * by comparing the current scores with the given one,
 * and if initial score value is too off, no result will
 * be enqueued.
 *
 */
template<class _Tp, unsigned _Dim, class _ComparatorType = std::less<real_t> >
class priority_queue {
public:
	typedef _ComparatorType comparator_type;
	typedef vectorx<_Tp, _Dim> point_type;
	typedef point_type &reference;
	typedef const point_type &const_reference;

private:
	real_t *_score;
	point_type *_data;
	unsigned _length;
	real_t _init_score;
	comparator_type _cmp_model;

	void allocate(unsigned length) {
		if (length > 0) {
			this->_score = new real_t[length];
			this->_data = new point_type[length];
		} else {
			this->_score = nullptr;
			this->_data = nullptr;
		}
		this->_lenght = length;
	}

	void deallocate() {
		delete [] this->_score;
		delete [] this->_data;
		this->_score = nullptr;
		this->_data = nullptr;
		this->_length = 0;
	}

	void init_score() {
		if (this->_length) {
			std::fill(this->_score, this->_score + this->_length, this->_init_score);
		}
	}

	void copy_score(real_t *score, unsigned length) {
		ASSERT(score);
		if (length) {
			if (length != this->_length)
				this->allocate(length);
			std::copy(score, score + length, _score);
		}
	}

public:
	//! Default constructor.
	priority_queue() :
		_score(nullptr), _data(nullptr), _length(0), _init_score(CV_BPQ_INIT_VALUE) {
	}
	//! Class constructor using size and initial value.
	priority_queue(unsigned length, real_t _init_score = CV_BPQ_INIT_VALUE) :
		_score(nullptr), _data(nullptr), _init_score(_init_score) 
	{
		this->allocate(length);
		init_score();
	}
	//! Copy constructor.
	priority_queue(const priority_queue& cpy) :
		_score(nullptr), _data(nullptr), _length(0), _init_score(cpy._init_score) 
	{
		this->allocate(cpy._length);
		std::copy(cpy._score, cpy._score + cpy._length, this->_score);
		std::copy(cpy._data, cpy._data + cpy._length, this->_data);
	}
	//! Move constructor.
	priority_queue(priority_queue &&move) :
	_score(move._score), _data(move._data), _length(move._length), _init_score(move._init_score) {
		move._score = nullptr;
		move._data = nullptr;
		move._length = 0;
		move._init_score = CV_BPQ_INIT_VALUE;
	}
	//! Class destructor.
	~priority_queue() {
	}

	//! Create the queue of given size, and initial score.
	priority_queue &create(unsigned length, real_t _init_score = CV_BPQ_INIT_VALUE) {

		ASSERT(length);

		this->_init_score = _init_score;
		this->deallocate();
		this->allocate(length);
		this->init_score();

		return *this;
	}

	//! Assignment operator.
	priority_queue &operator=(const priority_queue &rhs) {
		if (this != &rhs) {
			this->deallocate();
			this->allocate(rhs._length);
			this->_init_score = rhs._init_score;
			if (this->_score) {
				std::copy(rhs._score, rhs._score + rhs._length, this->_score);
				std::copy(rhs._data, rhs._data + rhs._length, this->_data);
			}
		}
		return *this;
	}

	//! Move operator.
	priority_queue &operator=(priority_queue &&rhs) {
		if (this != &rhs) {
			this->_score = rhs._score;
			this->_data = rhs._data;
			this->_length = rhs._length;
			this->_init_score = rhs._init_score;
			rhs._score = nullptr;
			rhs._data = nullptr;
			rhs._length = 0;
			rhs._init_score = CV_BPQ_INIT_VALUE;
		}
		return *this;
	}

	//! Append item to queue.
	void enque(const point_type &in, real_t score) {
		ASSERT(*this && score >= 0);

		for(unsigned i = 0; i < this->length(); ++i) {
			if (this->_cmp_model(score, this->_score[i])) {
				// move _data and _score, from i-th index to end.
				for (int j = this->length() - 1; j > i; j--) {
					this->_data[j] = this->_data[j - 1];
					this->_score[j] = this->_score[j - 1];
				}
				// set result
				this->_data[i] = in;
				_score[i] = score;
				return;
			}
		}
	}

	//! Return the value of the queue item at index.
	const_reference get_value(unsigned index) const {
		ASSERT(index < this->length());
		return this->_data[index];
	}

	//! Return the score of the queue item at index.
	real_t get_score(unsigned index) const {
		ASSERT(index < this->length());
		return this->_score[index];
	}

	//! Assign initialization score value.
	void set_init_score(real_t init) {
		this->_init_score = init;
	}

	/*!
	 * @brief Find out if indexed queue item has been populated with valid data.
	 *
	 * Non-populated queue item is the one with the score value
	 * equal to the initial score value.
	 */
	bool is_populated(unsigned index) const {
		ASSERT(index < this->length());
		return (_score[index] < _init_score) ? true : false;
	}

	//! Maximal distance in the queue, without including items which are not queued.
	real_t max_distance() {
		real_t dist = 0;
		for(unsigned i = 0; i < this->length(); ++i) {
			if (this->_score[i] < _init_score && this->_score[i] > dist)
				dist = this->_score[i];
		}
		return dist;
	}
};

}

#endif /* end of include guard: BPG_HPP_2QLEMDVV */