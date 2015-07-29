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


#include "../include/indexarray.hpp"

#include <cstring>
#include <algorithm>
#include <stdexcept>


namespace cv {

index_array::index_array() : _size(INDEX_ARRAY_SIZE) {
}

index_array::index_array(unsigned size) : _size(size) {
    ASSERT(size <= INDEX_ARRAY_SIZE);
}

index_array::index_array(const std::initializer_list<unsigned> &list) {
    ASSERT(list.size() <= INDEX_ARRAY_SIZE);
    this->_size = list.size();
    std::copy(list.begin(), list.end(), _data);
}

unsigned index_array::size() const {
    return this->_size;
}

bool index_array::empty() const {
    return this->_size == 0;
}

unsigned &index_array::operator[](unsigned i) {
    ASSERT(i < this->_size);
    return this->_data[i];
}

unsigned index_array::operator[](unsigned i) const {
    ASSERT(i < this->_size);
    return this->_data[i];
}

unsigned *index_array::begin() {
    return this->_data;
}

unsigned *index_array::end() {
    return this->_data + this->_size;
}

const unsigned *index_array::begin() const {
    return this->_data;
}

const unsigned *index_array::end() const {
    return this->_data + this->_size;
}

unsigned index_array::sum() const {
    long unsigned s = 0;
    for (auto v : *this) {
        s += v;
    }
    return s;
}

unsigned index_array::product() const {
    if (this->empty())
        return 0;
    long unsigned p = this->_data[0];
    for (unsigned i = 1; i < this->_size; ++i) {
        p *= this->_data[i];
    }
    return p;
}

void index_array::resize(unsigned size) {
    ASSERT(size <= INDEX_ARRAY_SIZE);
    this->_size = size;
}

void index_array::clear() {
    this->_size = 0;
    std::memset(this->_data, 0, sizeof(unsigned)*INDEX_ARRAY_SIZE);
}

bool index_array::operator==(const index_array &rhs) const {
    if (this->size() != rhs.size())
        return false;
    for (unsigned i = 0; i < this->size(); ++i) {
        if ((*this)[i] != rhs[i])
            return false;
    }
    return true;
}

bool index_array::operator!=(const index_array &rhs) const {
    return !operator==(rhs);
}

bool index_array::operator<(const index_array &rhs) const {
    return std::lexicographical_compare(this->begin(), this->end(), rhs.begin(), rhs.end());
}

index_array index_array::operator+(const index_array& rhs) const {
    ASSERT(this->size() == rhs.size());
    auto ret(*this);
    for (unsigned i = 0; i < this->size(); ++i) {
        ret[i]+= rhs[i];
    }
    return ret;
}

index_array index_array::operator-(const index_array& rhs) const {
    ASSERT(this->size() == rhs.size());
    auto ret(*this);
    for (unsigned i = 0; i < this->size(); ++i) {
        ret[i]-= rhs[i];
    }
    return ret;
}

index_array index_array::operator/(const index_array& rhs) const {
    ASSERT(this->size() == rhs.size());
    auto ret(*this);
    for (unsigned i = 0; i < this->size(); ++i) {
        ret[i]/= rhs[i];
    }
    return ret;
}

index_array index_array::operator*(const index_array& rhs) const {
    ASSERT(this->size() == rhs.size());
    auto ret(*this);
    for (unsigned i = 0; i < this->size(); ++i) {
        ret[i]*= rhs[i];
    }
    return ret;
}

index_array &index_array::operator+=(const index_array& rhs) {
    ASSERT(this->size() == rhs.size());
    for (unsigned i = 0; i < this->size(); ++i) {
        (*this)[i] += rhs[i];
    }
    return *this;
}

index_array &index_array::operator-=(const index_array& rhs) {
    ASSERT(this->size() == rhs.size());
    for (unsigned i = 0; i < this->size(); ++i) {
        (*this)[i] -= rhs[i];
    }
    return *this;
}

index_array &index_array::operator/=(const index_array& rhs) {
    ASSERT(this->size() == rhs.size());
    for (unsigned i = 0; i < this->size(); ++i) {
        (*this)[i] /= rhs[i];
    }
    return *this;
}

index_array &index_array::operator*=(const index_array& rhs) {
    ASSERT(this->size() == rhs.size());
    for (unsigned i = 0; i < this->size(); ++i) {
        (*this)[i] *= rhs[i];
    }
    return *this;
}

index_array index_array::operator+(unsigned rhs) const {
    auto ret(*this);
    for (auto &v : ret) {
        v += rhs;
    }
    return ret;
}

index_array index_array::operator-(unsigned rhs) const {
    auto ret(*this);
    for (auto &v : ret) {
        v -= rhs;
    }
    return ret;
}

index_array index_array::operator/(unsigned rhs) const {
    auto ret(*this);
    for (auto &v : ret) {
        v /= rhs;
    }
    return ret;
}

index_array index_array::operator*(unsigned rhs) const {
    auto ret(*this);
    for (auto &v : ret) {
        v *= rhs;
    }
    return ret;
}

index_array &index_array::operator+=(unsigned rhs) {
    for (auto &v : *this) {
        v += rhs;
    }
    return *this;
}

index_array &index_array::operator-=(unsigned rhs) {
    for (auto &v : *this) {
        v += rhs;
    }
    return *this;
}

index_array &index_array::operator/=(unsigned rhs) {
    for (auto &v : *this) {
        v += rhs;
    }
    return *this;
}

index_array &index_array::operator*=(unsigned rhs) {
    for (auto &v : *this) {
        v += rhs;
    }
    return *this;
}

}
