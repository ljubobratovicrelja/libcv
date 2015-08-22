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
// Implementation of contour structure, and various derivatives as a polygon
// structure.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef CONTOUR_HPP_XOOB1YC7
#define CONTOUR_HPP_XOOB1YC7

#include "fwd.hpp"
#include "vector.hpp"
#include "region.hpp"

#include <vector>

namespace cv {


/*!
 * @brief 2D contour template class.
 */
template<typename _Tp>
class contour {
  public:
	typedef _Tp value_type;
	typedef _Tp &reference;
	typedef const _Tp &const_reference;
	typedef vectorx<_Tp, 2> point_type;
	typedef vectorx<_Tp, 2> &point_reference;
	typedef const vectorx<_Tp, 2> &point_const_reference;

	typedef typename std::vector<vectorx<_Tp, 2> >::iterator iterator;
	typedef typename std::vector<vectorx<_Tp, 2> >::const_iterator const_iterator;


  protected:
	std::vector<point_type> _pts; // point array which defines the contour in index ordered manor.

  public:
	//! Default constructor.
	contour();
	//! Constructor using initial number of points in contour.
	contour(unsigned point_length);
	//! Constructor using range of points.
	template<typename point_iterator>
	contour(point_iterator begin, point_iterator end);
	//! Constructor using initializer list of points.
	contour(std::initializer_list<point_type> list);
	//! Copy constructor.
	contour(const contour &cpy);
	//! Move constructor.
	contour(contour &&move);
	//! Assignment operator - performs copy.
	contour &operator =(const contour &rhs);
	//! Move operator.
	contour &operator =(contour && rhs);
	//! Get point iterator at begin.
	iterator begin();
	//! Get point iterator at end.
	iterator end();
	//! Get read-only point iterator at begin.
	const_iterator begin() const;
	//! Get read-only point iterator at end.
	const_iterator end() const;
	//! Add point to the contour.
	void add_point(point_const_reference ptn);
	//! Add multiple points to the contour.
	void add_points(const std::vector<point_type> &ptns);
	//! Remove point from the contour.
	void remove_point(unsigned index);
	//! Remove all points, from the contour.
	void remove_all_points();
	//! Contour point getter.
	point_reference get_point(unsigned index);
	//! Contour read-only point getter.
	point_const_reference get_point(unsigned index) const;
	//! Get vector between two points in contour.
	vec2r get_edge_vector(unsigned startId, unsigned endId) const;
	//! Get vector between start and end points in the contour.
	vec2r get_contour_vector() const;
	//! Get count of points in countour.
	unsigned point_length() const;

	//! Check if contour object is empty of points.
	bool empty() const;

	//! Rearange points in contour by sorting them by given axis.
	void sort_by_axis(unsigned axis);

	//! Get bounding box of the contour.
	region<_Tp> get_bounding_box() const;

	//! Range operator - get part of contour. Creates a copy from begin to end.
	contour operator()(iterator begin, iterator end) const;

	//! Index operator for points in contour.
	point_reference operator[](unsigned index);
	//! Read-only index operator for points in contour.
	point_const_reference operator[](unsigned index) const;

	//! Add point to end of contour, as in push_back method.
	contour &operator <<(const vectorx<_Tp, 2> &ptn);
	//! Add multiple points to end of contour.
	contour &operator <<(const std::vector<vectorx<_Tp, 2> > &ptn);

	//! Equality operator, checks if all points are in the same position.
	bool operator ==(const contour &rhs) const;
	//! Inequality operator, checks if all points are in the same position.
	bool operator !=(const contour &rhs) const;
	//! Lexicographical compare of points in the contour.
	bool operator <(const contour &rhs) const;
	//! Check if there are any points in the contour.
	operator bool() const;
};

/*!
 * @brief 2D polygon template class.
 */
template<typename _Tp>
class polygon: public contour<_Tp> {
  private:
	typedef contour<_Tp> super;
  public:
	typedef typename contour<_Tp>::value_type value_type;
	typedef typename contour<_Tp>::reference reference;
	typedef typename contour<_Tp>::const_reference const_reference;
	typedef vectorx<_Tp, 2> point_type;
	typedef vectorx<_Tp, 2> &point_reference;
	typedef const vectorx<_Tp, 2> &point_const_reference;

	typedef typename std::vector<vectorx<_Tp, 2> >::iterator iterator;
	typedef typename std::vector<vectorx<_Tp, 2> >::const_iterator const_iterator;

	enum PointInPolygonStat { // TODO: point_in_polygon still does not utilize these - to be implemented.
		POINT_OUT_OF_POLYGON = 0, POINT_IN_POLYGON = 1, POINT_ON_EDGE = 2
	};

  public:
	//! Default constructor.
	polygon();
	//! Construct polygon with given number of points. All points are initialized to zero.
	polygon(unsigned point_length);
	//! Initialize polygon from initializer list of points.
	polygon(std::initializer_list<point_type> list);
	//! Copy constructor.
	polygon(const polygon &) = default;
	//! Move constructor.
	polygon(polygon &&) = default;

	//! Assignment operator.
	polygon &operator=(const polygon &) = default;
	//! Move operator.
	polygon &operator=(polygon &&) = default;

	//! Query if point is in polygon.
	unsigned point_in_polygon(point_const_reference ptn) const;
	//! Query if point is in polygon.
	unsigned point_in_polygon(const_reference x, const_reference y) const;
};

typedef contour<float> contourf;
typedef contour<double> contourd;
typedef contour<int> contouri;
typedef contour<short> contours;
typedef polygon<float> polygonf;
typedef polygon<double> polygond;
typedef polygon<int> polygoni;
typedef polygon<short> polygons;

#ifdef CV_REAL_TYPE_DOUBLE
typedef contourd countourr;
typedef polygond polygonr;
#else
typedef contourf countourr;
typedef polygonf polygonr;
#endif


// contour implementation ////////////////////////////////////////

template<typename _Tp>
contour<_Tp>::contour() {
}

template<typename _Tp>
contour<_Tp>::contour(unsigned point_length) :
	_pts(point_length) {
}
//! Constructor using range of points.
template<typename _Tp>
template<typename point_iterator>
contour<_Tp>::contour(point_iterator begin, point_iterator end) :
	_pts(begin, end) {
}

//! Constructor using initializer list of points.
template<typename _Tp>
contour<_Tp>::contour(std::initializer_list<point_type> list) :
	_pts(list) {
}
//! Copy constructor.
template<typename _Tp>
contour<_Tp>::contour(const contour &cpy) :
	_pts(cpy._pts) {
}

template<typename _Tp>
contour<_Tp>::contour(contour &&move) :
	_pts(std::move(move._pts)) {
}

template<typename _Tp>
contour<_Tp> &contour<_Tp>::operator =(const contour<_Tp> &rhs) {
	if (this != &rhs) {
		this->_pts = rhs._pts;
	}
	return *this;
}

template<typename _Tp>
contour<_Tp> &contour<_Tp>::operator=(contour<_Tp> && rhs) {
	if (this != &rhs) {
		this->_pts = std::move(rhs._pts);
	}
	return *this;
}

template<typename _Tp>
typename contour<_Tp>::iterator contour<_Tp>::begin() {
	return this->_pts.begin();
}

template<typename _Tp>
typename contour<_Tp>::iterator contour<_Tp>::end() {
	return this->_pts.end();
}

template<typename _Tp>
typename contour<_Tp>::const_iterator contour<_Tp>::begin() const {
	return this->_pts.begin();
}

template<typename _Tp>
typename contour<_Tp>::const_iterator contour<_Tp>::end() const {
	return this->_pts.end();
}

template<typename _Tp>
void contour<_Tp>::add_point(point_const_reference ptn) {
	this->_pts.push_back(ptn);
}

template<typename _Tp>
void contour<_Tp>::add_points(const std::vector<point_type> &ptns) {
	for (auto i : ptns) {
		this->_pts.push_back(i);
	}
}

template<typename _Tp>
void contour<_Tp>::remove_point(unsigned index) {
	ASSERT(index < this->_pts.size());
	this->_pts.remove(this->_pts.begin() + index);
}

template<typename _Tp>
void contour<_Tp>::remove_all_points() {
	this->_pts.clear();
}

template<typename _Tp>
typename contour<_Tp>::point_reference contour<_Tp>::get_point(unsigned index) {
	ASSERT(index < this->_pts.size());
	return this->_pts[index];
}

template<typename _Tp>
typename contour<_Tp>::point_const_reference contour<_Tp>::get_point(unsigned index) const {
	ASSERT(index < this->_pts.size());
	return this->_pts[index];
}

template<typename _Tp>
vec2r contour<_Tp>::get_edge_vector(unsigned startId, unsigned endId) const {
	ASSERT(startId < this->_pts.size() && endId < this->_pts.size() && startId < endId);
	return (this->_pts[endId] - this->_pts[startId]);
}

template<typename _Tp>
vec2r contour<_Tp>::get_contour_vector() const {
	return this->get_edge_vector(0, this->point_length() - 1);
}

template<typename _Tp>
unsigned contour<_Tp>::point_length() const {
	return this->_pts.size();
}

template<typename _Tp>
bool contour<_Tp>::empty() const {
	return this->point_length() == 0;
}

template<typename _Tp>
void contour<_Tp>::sort_by_axis(unsigned axis) {
	ASSERT(axis == 0 || axis == 1);
	std::sort(this->begin(), this->end(), internal::idx_cmp(axis));
}

template<typename _Tp>
region<_Tp> contour<_Tp>::get_bounding_box() const {
	point_type min, max;

	auto data = this->_pts.data();
	bool init = false;

	for (int i = 0; i < this->point_length(); ++i) {
		if (!init) {
			min = data[i];
			max = data[i];
			init = true;
			continue;
		}

		if (data[i][0] < min[0]) {
			min[0] = data[i][0];
		}
		if (data[i][1] < min[1]) {
			min[1] = data[i][1];
		}
		if (data[i][0] > max[0]) {
			max[0] = data[i][0];
		}
		if (data[i][1] > max[1]) {
			max[1] = data[i][1];
		}
	}
	return region<_Tp>(min[0], min[1], max[0] - min[0], max[1] - min[1]);
}

template<typename _Tp>
contour<_Tp> contour<_Tp>::operator()(iterator begin, iterator end) const {
	return contour(begin, end);
}

template<typename _Tp>
typename contour<_Tp>::point_reference contour<_Tp>::operator[](unsigned index) {
	ASSERT(index < this->_pts.size());
	return this->_pts[index];
}

template<typename _Tp>
typename contour<_Tp>::point_const_reference contour<_Tp>::operator[](unsigned index) const {
	ASSERT(index < this->_pts.size());
	return this->_pts[index];
}

template<typename _Tp>
contour<_Tp> &contour<_Tp>::operator <<(const vectorx<_Tp, 2> &ptn) {
	this->_pts.push_back(ptn);
	return *this;
}

template<typename _Tp>
contour<_Tp> &contour<_Tp>::operator <<(const std::vector<vectorx<_Tp, 2> > &ptn) {
	this->add_points(ptn);
	return *this;
}

template<typename _Tp>
bool contour<_Tp>::operator ==(const contour &rhs) const {
	if (this->point_length() != rhs.point_length()) {
		return false;
	}
	for (int i = 0; i < this->point_length(); ++i) {
		if (this->_pts[i] != rhs[i]) {
			return false;
		}
	}
	return true;
}

template<typename _Tp>
bool contour<_Tp>::operator!=(const contour &rhs) const {
	return !operator ==(rhs);
}

template<typename _Tp>
bool contour<_Tp>::operator <(const contour &rhs) const {
	return std::lexicographical_compare(this->begin(), this->end(), rhs.begin(), rhs.end());
}

template<typename _Tp>
contour<_Tp>::operator bool() const {
	return static_cast<bool>(this->_pts.length());
}


// polygon implementation ////////////////////////////////////////

template<typename _Tp>
polygon<_Tp>::polygon() :
	super() {
}

template<typename _Tp>
polygon<_Tp>::polygon(unsigned point_length) :
	super(point_length) {
}

template<typename _Tp>
polygon<_Tp>::polygon(std::initializer_list<point_type> list) :
	super(list) {
}

template<typename _Tp>
unsigned polygon<_Tp>::point_in_polygon(point_const_reference ptn) const {
	return point_in_polygon(ptn[0], ptn[1]);
}

template<typename _Tp>
unsigned polygon<_Tp>::point_in_polygon(const_reference x, const_reference y) const {
	unsigned i, j, c = 0;
	auto pt_data = this->_pts.data();
	for (i = 0, j = this->point_length() - 1; i < this->point_length(); j = i++) {
		if (((pt_data[i][1] > y) != (pt_data[j][1] > y))
		        && (x
		            < (pt_data[j][0] - pt_data[i][0]) * (y - pt_data[i][1]) / (pt_data[j][1] - pt_data[i][1])
		            + pt_data[i][0])) {
			c = !c;
		}
	}
	return c;
}

}

#endif /* end of include guard: CONTOUR_HPP_XOOB1YC7 */






