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
// Module with basic shape drawing functions.
// 
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef DRAW_HPP_TO7EN2ZF
#define DRAW_HPP_TO7EN2ZF


#include "vector.hpp"
#include "contour.hpp"
#include "matrix.hpp"
#include "matfunc.hpp"
#include "region.hpp"


namespace cv {

template<typename _Tp>
void draw_point(matrix<_Tp>& mat, const vec2i& point, const _Tp &color, unsigned strokeWidth = 1) {
	ASSERT(mat && strokeWidth > 0);

	if (strokeWidth == 1) {
		if (point[1] >= 0 && point[1] < mat.rows() && point[0] >= 0 && point[0] < mat.cols()) {
			mat(point[1], point[0]) = color;
		}
	} else {
		for (unsigned r = point[1] - strokeWidth / 2; r < point[1] + strokeWidth / 2; r++) {
			for (unsigned c = point[0] - strokeWidth / 2; c < point[0] + strokeWidth / 2; c++) {
				if (r >= 0 && r < mat.rows() && c >= 0 && c < mat.cols())
					mat(r, c) = color;
			}
		}
	}
}

template<typename _Tp>
void draw_line(matrix<_Tp>& mat, const vec2i& startPoint, const vec2i& endPoint, const _Tp & color, unsigned strokeWidth = 1) {

	ASSERT(mat && strokeWidth > 0);

	int x_length = abs(endPoint[0] - startPoint[0]);
	int y_length = abs(endPoint[1] - startPoint[1]);

	int longer_axis;

	vec2r line_vec(endPoint[0] - startPoint[0], endPoint[1] - startPoint[1]), norm_vec;
	vec2r curr_point(startPoint[0], startPoint[1]), ptn;

	// rotate the line_vec for 90 degrees.
	norm_vec = vec2r{(real_t)(line_vec[0] * cos(PI/2.) - line_vec[1] * sin(PI/2.)), 
					(real_t)(line_vec[0] * sin(PI/2.) + line_vec[1] * cos(PI/2.))};
	norm_vec /= norm_vec.sum();

	if (x_length >= y_length) {
		longer_axis = x_length;
	} else {
		longer_axis = y_length;
	}

	line_vec /= longer_axis;

	int st = strokeWidth;
	for (int i = 0; i < longer_axis; i++) {
		if (st > 1) {
			for (real_t j = -1 * st / 2; j < st / 2; j += 0.1) {
				ptn = curr_point + (norm_vec * j);
				draw_point(mat, vec2i(ptn[0], ptn[1]), color);
			}
		} else {
			draw_point(mat, vec2i(curr_point[0], curr_point[1]), color);
		}
		curr_point += line_vec;
	}
}

template<typename _Tp>
void draw_circle(matrix<_Tp>& mat, const vec2i& center, unsigned radius, const _Tp & color, unsigned strokeWidth = 1,
		unsigned interpolationStep = 24) {
	ASSERT(mat);

	vec2r radius_vec;

	vec2r curr_ptn = {static_cast<real_t>(center[0] + radius*cos(0.)), static_cast<real_t>(center[1] + radius*sin(0.))};
	vec2r next_ptn;

	real_t draw_step = static_cast<real_t>((2. * PI)) / interpolationStep;

	real_t ang = 0.0;
	for (unsigned i = 0; i < interpolationStep+1; ++i) {
		next_ptn[0] = center[0] + (radius) * cos(ang + draw_step);
		next_ptn[1] = center[1] + (radius) * sin(ang + draw_step);
		draw_line(mat, curr_ptn, next_ptn, color, strokeWidth);
		curr_ptn = next_ptn;
		ang += draw_step;
	}
}

template<typename _Tp>
void draw_contour(matrix<_Tp>& mat, const contouri &contour, const _Tp & color, unsigned strokeWidth = 1) {
	for (unsigned i = 1; i < contour.point_length(); ++i) {
		draw_line(mat, contour[i - 1], contour[i], color, strokeWidth);
	}
}

template<typename _Tp>
void draw_polygon(matrix<_Tp> &mat, const polygoni &polygon, const _Tp &color, unsigned strokeWidth = 1, bool filled = false) {

	if (filled) {
		auto bb = polygon.get_bounding_box();
		for(unsigned i = bb.x; i < bb.x + bb.width; ++i) {
			for(unsigned j = bb.y; j < bb.y + bb.height; ++j) {
				auto res = polygon.point_in_polygon({static_cast<int>(i), static_cast<int>(j)});
				if (res) {
					mat(j, i) = color;
				}
			}
		}
	} else {
		for (unsigned i = 0; i < polygon.point_length(); ++i) {
			draw_line(mat, polygon[i - 1], polygon[i], color, strokeWidth);
		}
		draw_line(mat, polygon[polygon.point_length() - 1], polygon[0], color, strokeWidth);
	}
}

template<typename _Tp>
void draw_rect(matrix<_Tp>& mat, const regioni& reg, const _Tp & color, unsigned strokeWidth) {

	regioni dReg = reg;

	if (dReg.x < 0)
		dReg.x = 0;

	if (dReg.y < 0)
		dReg.y = 0;

	if (dReg.x + dReg.width >= mat.cols())
		dReg.width = (dReg.x + dReg.width) - mat.cols();
	if (dReg.y + dReg.height >= mat.rows())
		dReg.height = (dReg.y + dReg.height) - mat.rows();

	draw_line(mat, vec2i(dReg.x, dReg.y), vec2i(dReg.x + dReg.width, dReg.y), color, strokeWidth);
	draw_line(mat, vec2i(dReg.x, dReg.y), vec2i(dReg.x, dReg.y + dReg.height), color, strokeWidth);
	draw_line(mat, vec2i(dReg.x + dReg.width, dReg.y), vec2i(dReg.x + dReg.width, dReg.y + dReg.height), color, strokeWidth);
	draw_line(mat, vec2i(dReg.x, dReg.y + dReg.height), vec2i(dReg.x + dReg.width, dReg.y + dReg.height), color, strokeWidth);
}


}

#endif /* end of include guard: DRAW_HPP_TO7EN2ZF */
