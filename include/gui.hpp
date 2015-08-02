//The MIT License (MIT)
//
//Copyright (c) 2015 Relja Ljubobratovic, ljubobratovic.relja@gmail.com
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.
//
// Description:
// Contains minimalistic GUI library for viewing images, formated as
// cv::image_array from image.hpp. Uses Qt library.
//
// Note:
// This module can be disabled ad compile time with a cmake (compilation) 
// flag 'CV_IGNORE_GUI'.
// Main purpose of it is to debug algorithms while in development. For more
// complete gui solution, it would be required to extend it with another Qt
// solution.
//
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com

#ifndef CV_INGORE_GUI // ignore includes to another sources.

#ifndef GUI_HPP_I1AGFQQP
#define GUI_HPP_I1AGFQQP


#include "fwd.hpp"
#include "image.hpp"


namespace cv {

void CV_EXPORT imshow(const std::string &name, const image_array &image);
void CV_EXPORT imclose(const std::string &name);
void CV_EXPORT imclose_all();
char CV_EXPORT wait_key();

}

#endif // CV_IGNORE_GUI
#endif /* end of include guard: GUI_HPP_I1AGFQQP */
