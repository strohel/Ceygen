/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

#include <Eigen/Core>

using namespace Eigen;

template<typename dtype>
class VectorMap : public Map<Matrix<dtype, Dynamic, 1> >
{
	public:
		VectorMap() : Map<Matrix<dtype, Dynamic, 1> >(0, 0) {};
		void init(dtype *data, int rows) {};
};

template<typename dtype>
class MatrixMap : public Map<Matrix<dtype, Dynamic, Dynamic> >
{
	public:
		MatrixMap() : Map<Matrix<dtype, Dynamic, Dynamic> >(0, 0, 0) {};
		void init(dtype *data, int rows, int cols) {};
};
