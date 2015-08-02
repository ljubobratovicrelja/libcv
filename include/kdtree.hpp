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
// K-dimentional tree structure implementation, with implemented methods for
// nearest neighbour search.
// 
// Author:
// Relja Ljubobratovic, ljubobratovic.relja@gmail.com


#ifndef KDTREE_HPP_XRTGWOEM
#define KDTREE_HPP_XRTGWOEM


#include <iostream>
#include <vector>

#include "fwd.hpp"
#include "vector.hpp"
#include "bpq.hpp"


namespace cv {

enum class kd_treeSearchType {
	VALUE,
	INDEX
};

template<class _Tp, int dims>
class kd_node;
template<class _Tp, int dims, class Comparator>
class kd_tree;

/*!
 * Default vector comparator.
 * Used to compare values
 * of a vector by a certain axis.
 */
template<class _Tp, int dims>
class kd_node_cmp {
private:
	int dim;  //!< Dimension used for comparison
public:
	//! Constructor using specified dimension.
	kd_node_cmp(int dim);
	//! Compare operator.
	bool operator()(kd_node<_Tp, dims> *rhs,
			kd_node<_Tp, dims> *lhs);
};

template<class _Tp, int dims, class Comparator = kd_node_cmp<_Tp, dims> > class kd_tree;

/*!
 * @brief Class defines k-dimension _tree node.
 */
template<class _Tp, int dims>
class kd_node {
	friend class kd_tree<_Tp, dims> ;
protected:
	unsigned _id;  //!< Index of the node.
	vectorx<_Tp, dims> _data;  //!< Data of the node.
	kd_tree<_Tp, dims>* _tree;  //!< Pointer to the _tree node.
	kd_node<_Tp, dims>* _parent;  //!< Parent of the node.
	kd_node<_Tp, dims>* _l_child;  //!< Left child of the node.
	kd_node<_Tp, dims>* _r_child;  //!< Right child of the node.

public:
	//! Class constructor.
	kd_node();
	//! Class constructor.
	kd_node(vectorx<_Tp, dims> data, unsigned id,
			kd_tree<_Tp, dims>* tree);
	//! Copy constructor.
	kd_node(const kd_node<_Tp, dims>& cpy);
	//! Assignment operator - does not share _data ownership, it copies it instead.
	kd_node<_Tp, dims>& operator=(const kd_node<_Tp, dims>& rhs);
	//! Class destructor.
	virtual ~kd_node();

	//! Create _tree node.
	void create(vectorx<_Tp, dims> data, unsigned id,
			kd_tree<_Tp, dims>* _tree);

	//! Set node index.
	void set_id(unsigned id);
	//! Set pointer to the node _data.
	void set_data(const vectorx<_Tp, dims>& data);
	//! Set pointer to the _tree.
	void set_tree(kd_tree<_Tp, dims>* tree);
	//! Get _data of the node.
	vectorx<_Tp, dims>& get_data();
	//! Get _data of the node.
	const vectorx<_Tp, dims>& get_data() const;
	//! Get node index.
	unsigned get_id() const;
	//! Assign _parent.
	void assign_parent(kd_node<_Tp, dims>* parent);
	//! Assign left child.
	void assign_left_child(kd_node<_Tp, dims>* child);
	//! Assign right child.
	void assign_right_child(kd_node<_Tp, dims>* child);
	//! Get kd_tree of this node
	kd_tree<_Tp, dims>* get_tree();

	//! Comparison operator.
	bool operator==(const kd_node& rhs) const;
	//! Negative comparison operator.
	bool operator!=(const kd_node& rhs) const;
};

/*!
 * @brief Class defines k-dimension binary _tree.
 *
 * K-dimensional _tree structure implementation.
 * Class is designed as template with predefined
 * explicit instantiations, which are typedef-d.
 */
template<class _Tp, int dims, class Comparator>
class kd_tree {
public:
	typedef kd_node<_Tp, dims> node;  //!< node type of the _tree.
	typedef vectorx<_Tp, dims> vec;  //! Type of vector used in the _tree.
	typedef std::vector<node*> node_list;

	typedef typename node_list::iterator iterator;
	typedef typename node_list::const_iterator const_iterator;


protected:
	std::vector<node*> _tree_nodes;  //!< Nodes of the _tree.
	std::vector<node*> _node_data; //! Data of nodes.

	//! Recursive _tree branching method.
	node* build_node_from_data(typename std::vector<node*>::iterator begin,
			typename std::vector<node*>::iterator end, uint depth = 0);
	//! Recursive neighbour of the node.
	void find_nn(const vectorx<_Tp, dims> &searchPoint, node *guess, node* curr,
			double &bestDist, int depth = 0);
	//! Find k neighours.
	void find_knn(const vectorx<_Tp, dims> &searchPoint, node *guess, node* curr,
			priority_queue<node*> &bpq, unsigned count, double &bestDist, int depth = 0);
	//! Clear _data of the kd-_tree.
	void clear_data();

public:
	//! Class constructor.
	kd_tree();
	//! Class constructor.
	kd_tree(const std::vector<vectorx<_Tp, dims> > &pointList);
	//! Copy constructor.
	kd_tree(const kd_tree&);
	//! Class destructor.
	virtual ~kd_tree();
	//! Assignment operator.
	kd_tree& operator=(const kd_tree&);

	//! Build this _tree with present _data.
	void build();
	//! Set data to this tree - this tree receives ownership of points. Requires rebuilding the tree after changing data set.
	void set_data(const std::vector<vectorx<_Tp, dims> > &data);
	//! Find nearest node for given point.
	vectorx<_Tp, dims> nn(const vectorx<_Tp, dims> &searchPoint);
	//! Find nearest neighours for given data.
	void knn(const vec& searchPoint, unsigned nCount,
			std::vector<vectorx<_Tp, dims> >& outData, double max_distance = 0.0);
	//! Find nearest neighours for given data.
	void knn_index(const vec& searchPoint, unsigned nCount, std::vector<unsigned>& outData, double max_distance = 0.0);
	//! Find nearest neighours for given data.
	void knn_tree_node(const vec& searchPoint, unsigned nCount,
			std::vector<node*>& outData, double max_distance = 0.0);
	//! Get root node.
	kd_node<_Tp, dims>* get_root_node();
	//! Get root node.
	const kd_node<_Tp, dims>* get_root_node() const;
	//! Get node.
	const kd_node<_Tp, dims>* get_node(unsigned index) const;

	iterator begin() {
		return _tree_nodes.begin();
	}

	iterator end() {
		return _tree_nodes.end();
	}

	const_iterator begin() const {
		return _tree_nodes.begin();
	}

	const_iterator end() const {
		return _tree_nodes.end();
	}
};

typedef kd_tree<int, 2> kd_tree2i;
typedef kd_tree<float, 2> kd_tree2f;
typedef kd_tree<double, 2> kd_tree2d;
typedef kd_tree<byte, 2> kd_tree2b;

typedef kd_tree<int, 3> kd_tree3i;
typedef kd_tree<float, 3> kd_tree3f;
typedef kd_tree<double, 3> kd_tree3d;
typedef kd_tree<byte, 3> kd_tree3b;

typedef kd_tree<int, 4> kd_tree4i;
typedef kd_tree<float, 4> kd_tree4f;
typedef kd_tree<double, 4> kd_tree4d;
typedef kd_tree<byte, 4> kd_tree4b;

typedef kd_tree<int, 5> kd_tree5i;
typedef kd_tree<float, 5> kd_tree5f;
typedef kd_tree<double, 5> kd_tree5d;
typedef kd_tree<byte, 5> kd_tree5b;

typedef kd_tree<int, 6> kd_tree6i;
typedef kd_tree<float, 6> kd_tree6f;
typedef kd_tree<double, 6> kd_tree6d;
typedef kd_tree<byte, 6> kd_tree6b;

typedef kd_node<int, 2> kd_node2i;
typedef kd_node<float, 2> kd_node2f;
typedef kd_node<double, 2> kd_node2d;
typedef kd_node<byte, 2> kd_node2b;

typedef kd_node<int, 3> kd_node3i;
typedef kd_node<float, 3> kd_node3f;
typedef kd_node<double, 3> kd_node3d;
typedef kd_node<byte, 3> kd_node3b;

typedef kd_node<int, 4> kd_node4i;
typedef kd_node<float, 4> kd_node4f;
typedef kd_node<double, 4> kd_node4d;
typedef kd_node<byte, 4> kd_node4b;

typedef kd_node<int, 5> kd_node5i;
typedef kd_node<float, 5> kd_node5f;
typedef kd_node<double, 5> kd_node5d;
typedef kd_node<byte, 5> kd_node5b;

typedef kd_node<int, 6> kd_node6i;
typedef kd_node<float, 6> kd_node6f;
typedef kd_node<double, 6> kd_node6d;
typedef kd_node<byte, 6> kd_node6b;


//! Constructor using specified dimension.
template<class _Tp, int dims>
kd_node_cmp<_Tp, dims>::kd_node_cmp(int dim) {
    this->dim = dim;
}
//! Compare operator.
template<class _Tp, int dims>
bool kd_node_cmp<_Tp, dims>::operator()(kd_node<_Tp, dims> *rhs,
        kd_node<_Tp, dims> *lhs) {
    return (rhs->get_data()[dim] < lhs->get_data()[dim]);
}

///////////////////////////////////////////////////////////////////////////////
//
//	CLASS:
//	kd_node
//
//	DESCRIPTION:
//	node item registered by kd_tree class.
//
///////////////////////////////////////////////////////////////////////////////

template<class _Tp, int dims>
kd_node<_Tp, dims>::kd_node() :
    _id(0), _data(vectorx<_Tp, dims>()), _tree(nullptr), _parent(nullptr), _l_child(
        nullptr), _r_child(nullptr) {
}

template<class _Tp, int dims>
kd_node<_Tp, dims>::kd_node(vectorx<_Tp, dims> _data, unsigned _id,
                                  kd_tree<_Tp, dims>* _tree) {
    create(_data, _id, _tree);
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::create(vectorx<_Tp, dims> _data, unsigned _id,
                                   kd_tree<_Tp, dims>* _tree) {
    this->_id = _id;
    this->_tree = _tree;
    this->_data = _data;
    this->_parent = nullptr;
    this->_l_child = nullptr;
    this->_r_child = nullptr;
}

template<class _Tp, int dims>
kd_node<_Tp, dims>::kd_node(const kd_node& cpy) {
    _data = cpy._data;
    _id = cpy._id;
    _tree = cpy._tree;
    _parent = cpy._parent;
    _l_child = cpy._l_child;
    _r_child = cpy._r_child;
}

template<class _Tp, int dims>
kd_node<_Tp, dims>& kd_node<_Tp, dims>::operator=(const kd_node& rhs) {
    if (this != &rhs) {
        _data = rhs._data;
        _id = rhs._id;
        _tree = rhs._tree;
        _parent = rhs._parent;
        _l_child = rhs._l_child;
        _r_child = rhs._r_child;
    }
    return *this;
}

template<class _Tp, int dims>
kd_node<_Tp, dims>::~kd_node() {
    _tree = nullptr;
    _parent = nullptr;
    _l_child = nullptr;
    _r_child = nullptr;
}

template<class _Tp, int dims>
kd_tree<_Tp, dims>* kd_node<_Tp, dims>::get_tree() {
    return _tree;
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::set_id(unsigned _id) {
    this->_id = _id;
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::set_data(const vectorx<_Tp, dims>& _data) {
    this->_data = _data;
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::set_tree(kd_tree<_Tp, dims>* _tree) {
    this->_tree = _tree;
}

template<class _Tp, int dims>
vectorx<_Tp, dims>& kd_node<_Tp, dims>::get_data() {
    return _data;
}

template<class _Tp, int dims>
const vectorx<_Tp, dims>& kd_node<_Tp, dims>::get_data() const {
    return _data;
}

template<class _Tp, int dims>
unsigned kd_node<_Tp, dims>::get_id() const {
    return _id;
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::assign_parent(kd_node<_Tp, dims>* _parent) {
    this->_parent = _parent;
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::assign_left_child(kd_node<_Tp, dims>* child) {
    _l_child = child;
    if (child)
        child->_parent = this;
}

template<class _Tp, int dims>
void kd_node<_Tp, dims>::assign_right_child(kd_node<_Tp, dims>* child) {
    _r_child = child;
    if (child)
        child->_parent = this;
}

template<class _Tp, int dims>
bool kd_node<_Tp, dims>::operator==(const kd_node<_Tp, dims>& rhs) const {
    return (_data == rhs._data && _parent == rhs._parent
            && _l_child == rhs._l_child && _r_child == rhs._r_child
            && _tree == rhs._tree && _parent != nullptr && _l_child != nullptr
            && _r_child != nullptr && _tree != nullptr);
}

template<class _Tp, int dims>
bool kd_node<_Tp, dims>::operator!=(const kd_node& rhs) const {
    return !kd_node::operator==(rhs);
}

///////////////////////////////////////////////////////////////////////////////
//
//	CLASS:
//	kd_tree
//
//	DESCRIPTION:
//	k-dimensional _tree structure class.
//
///////////////////////////////////////////////////////////////////////////////

template<class _Tp, int dims, class Comparator>
kd_tree<_Tp, dims, Comparator>::kd_tree() {
}

template<class _Tp, int dims, class Comparator>
kd_tree<_Tp, dims, Comparator>::kd_tree(
    const std::vector<vectorx<_Tp, dims> > &pointList) {
    if (!pointList.empty()) {
        this->set_data(pointList);
        build();
    } else {
        std::cerr << "kd_tree construction error!~ (!pointList.empty()).\n";
    }
}

template<class _Tp, int dims, class Comparator>
kd_tree<_Tp, dims, Comparator>::kd_tree(const kd_tree& rhs) {
    _tree_nodes = rhs._tree_nodes;
}

template<class _Tp, int dims, class Comparator>
kd_tree<_Tp, dims, Comparator>::~kd_tree() {

}
template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::clear_data() {

    while (!_node_data.empty())
        delete &_node_data.back(), _node_data.pop_back();

    _tree_nodes.clear();
}

template<class _Tp, int dims, class Comparator>
kd_tree<_Tp, dims, Comparator>& kd_tree<_Tp, dims, Comparator>::operator=(
    const kd_tree& rhs) {
    if (this != &rhs) {
        _tree_nodes = rhs._tree_nodes;
    }
    return *this;
}

template<class _Tp, int dims, class Comparator>
kd_node<_Tp, dims>* kd_tree<_Tp, dims, Comparator>::build_node_from_data(
    typename std::vector<kd_node<_Tp, dims> *>::iterator begin,
    typename std::vector<kd_node<_Tp, dims> *>::iterator end, uint depth) {

    std::vector<kd_node<_Tp, dims> *> _data(begin, end);

    // if _data is empty return just null.
    if (_data.empty())
        return nullptr;

    // reset depth if reached end.
    if (depth == dims)
        depth = 0;

    // sort the given _data by the depth
    std::sort(_data.begin(), _data.end(), Comparator(depth));

    // get median value.
    int median = (_data.size() / 2);

    node *mNode = _data[median];
    _tree_nodes.push_back(mNode);

    _data.erase(_data.begin() + median);

    if (_data.size() > 0) {

        // get recursively left and righ child.
        kd_node<_Tp, dims> *_l_child = build_node_from_data(_data.begin(),
                                           _data.begin() + _data.size() / 2, depth + 1);
        kd_node<_Tp, dims> *_r_child = build_node_from_data(_data.begin() + _data.size() / 2,
                                            _data.end(), depth + 1);

        mNode->assign_left_child(_l_child);
        mNode->assign_right_child(_r_child);
    }

    return mNode;
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::build() {
    if(_node_data.empty()) {
        return;
    }
    _tree_nodes.clear();
    build_node_from_data(_node_data.begin(), _node_data.end()); // recursive building of the _tree.
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::set_data(
    const std::vector<vectorx<_Tp, dims> > &_data) {
    clear_data();
	for(auto v : _data) {
        _node_data.push_back(
            new kd_node<_Tp, dims>(v, _node_data.size(), this));
    }
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::find_nn(const vectorx<_Tp, dims> &searchPoint,
        node *guess, node* curr, double &bestDist, int depth) {
    if (_tree_nodes.empty())
        return;
    if (depth >= dims)
        depth = 0;

    if (!guess) {
        curr = _tree_nodes[0];
        guess = curr;
    } else {

        if (!curr)
            return;

        double currDist = searchPoint.distance(curr->get_data());

        if (currDist < bestDist) {
            bestDist = currDist;
            guess = curr;
        }
    }

    bool leftRight = (searchPoint[depth] < curr->get_data()[depth]);

    if (leftRight) {
        find_nn(searchPoint, guess, curr->_l_child, bestDist, depth + 1);
    } else {
        find_nn(searchPoint, guess, curr->_r_child, bestDist, depth + 1);
    }

    if (abs(curr->get_data()[depth] - searchPoint[depth]) < bestDist) {
        if (!leftRight)
            find_nn(searchPoint, guess, curr->_l_child, bestDist, depth + 1);
        else
            find_nn(searchPoint, guess, curr->_r_child, bestDist, depth + 1);
    }
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::find_knn(const vectorx<_Tp, dims> &searchPoint,
        node *guess, node* curr, priority_queue<node*> &bpq, unsigned count, double &bestDist,
        int depth) {
    if (_tree_nodes.empty())
        return;
    if (depth >= dims)
        depth = 0;

    if (!guess) {
        curr = _tree_nodes[0];
        guess = curr;
    } else {

        if (!curr)
            return;

        double currDist = searchPoint.distance(curr->get_data());

        bpq.enque(curr, currDist);

        if (currDist < bestDist) {
            bestDist = currDist;
            guess = curr;
        }
    }

    if (searchPoint[depth] < curr->get_data()[depth]) {
        find_knn(searchPoint, guess, curr->_l_child, bpq,count, bestDist, depth + 1);
        if (abs(curr->get_data()[depth] - searchPoint[depth]) < bpq.max_distance()) {
            find_knn(searchPoint, guess, curr->_r_child, bpq,count, bestDist,
                    depth + 1);
        }
    } else {
        find_knn(searchPoint, guess, curr->_r_child, bpq,count, bestDist, depth + 1);
        if (abs(curr->get_data()[depth] - searchPoint[depth]) < bpq.max_distance()) {
            find_knn(searchPoint, guess, curr->_l_child, bpq,count, bestDist,
                    depth + 1);
        }
    }
}

template<class _Tp, int dims, class Comparator>
vectorx<_Tp, dims> kd_tree<_Tp, dims, Comparator>::nn(
    const vectorx<_Tp, dims> &searchPoint) {
    ASSERT(_tree_nodes.size());
    node* node = nullptr;
    double bestDist = 99999999.999;
    find_nn(searchPoint, node, _tree_nodes[0], bestDist);

    if (!node)
        throw std::runtime_error("Error finding nearest node.");

    return node->get_data();
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::knn(const vec& searchPoint, unsigned nCount,
                                        std::vector<vec>& outData, double max_distance) {
    ASSERT(nCount > 0 && max_distance >= 0.0 && _tree_nodes.size() != 0);
    outData.clear();

    priority_queue<node*> bpq(nCount);

    double bestDist = 99999999.999;
    find_knn(searchPoint, _tree_nodes[0], _tree_nodes[0], bpq, nCount, bestDist);

    LOOP_FOR_TO(nCount)
    {
        if (bpq.isPopulated(i)) {
            if (max_distance > 0.0) {
                if (bpq.get_score(i) < max_distance)
                    outData.push_back(bpq.get_value(i)->get_data());
            } else {
                outData.push_back(bpq.get_value(i)->get_data());
            }
        }
    }
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::knn_index(const vec& searchPoint, unsigned nCount,
        std::vector<unsigned>& outData, double max_distance) {
    ASSERT(nCount > 0 && max_distance >= 0.0 && _tree_nodes.size() != 0);
    outData.clear();

    priority_queue<node*> bpq(nCount);

    double bestDist = 99999999.999;
    find_knn(searchPoint, _tree_nodes[0], _tree_nodes[0], bpq, nCount, bestDist);

	for (unsigned i = 0; i < nCount; ++i) {
        if (bpq.is_populated(i)) {
            if (max_distance > 0.0) {
                if (bpq.get_score(i) < max_distance)
                    outData.push_back(bpq.get_value(i)->get_id());
            } else {
                outData.push_back(bpq.get_value(i)->get_id());
            }
        }
    }
}

template<class _Tp, int dims, class Comparator>
void kd_tree<_Tp, dims, Comparator>::knn_tree_node(const vec& searchPoint, unsigned nCount,
        std::vector<node*>& outData, double max_distance) {
    ASSERT(nCount > 0 && max_distance >= 0.0 && _tree_nodes.size() != 0);
    outData.clear();

    priority_queue<node*> bpq(nCount);

    double bestDist = 99999999.999;
    find_knn(searchPoint, _tree_nodes[0], _tree_nodes[0], bpq, nCount, bestDist);

    LOOP_FOR_TO(nCount)
    {
        if (bpq.isPopulated(i)) {
            if (max_distance > 0.0) {
                if (bpq.get_score(i) < max_distance)
                    outData.push_back(bpq.get_value(i));
            } else {
                outData.push_back(bpq.get_value(i));
            }
        }
    }
}
template<class _Tp, int dims, class Comparator>
kd_node<_Tp, dims>* kd_tree<_Tp, dims, Comparator>::get_root_node() {
    return ((_tree_nodes.size()) ? _tree_nodes[0] : nullptr);
}

template<class _Tp, int dims, class Comparator>
const kd_node<_Tp, dims>* kd_tree<_Tp, dims, Comparator>::get_root_node() const {
    return get_node(0);
}

template<class _Tp, int dims, class Comparator>
const kd_node<_Tp, dims>* kd_tree<_Tp, dims, Comparator>::get_node(
    unsigned index) const {
    return (_tree_nodes.size() == 0 || index >= _tree_nodes.size()) ?
           nullptr : _tree_nodes[index];
}

}

#endif /* end of include guard: KDTREE_HPP_XRTGWOEM */
