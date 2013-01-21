/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

#include <stdexcept>
// make Eigen raise an exception instead of aborting on assert failure. Cython converts
// std::runtime_error to Python RuntimeError
#define eigen_assert(statement) do { if(!(statement)) throw std::runtime_error(#statement " does not hold (in Eigen)"); } while(0)

#include <Eigen/Core>

using namespace Eigen;

template<typename dtype>
class VectorMap : public Map<Matrix<dtype, Dynamic, 1> >
{
	public:
		VectorMap() : Map<Matrix<dtype, Dynamic, 1> >(0, 0) {}
		void init(dtype *data, int rows) {
			// see http://eigen.tuxfamily.org/dox/TutorialMapClass.html
			// this is NOT a heap allocation:
			new (this) Map<Matrix<dtype, Dynamic, 1> >(data, rows);
		};
};

template<typename dtype>
class MatrixMap : public Map<Matrix<dtype, Dynamic, Dynamic> >
{
	public:
		MatrixMap() : Map<Matrix<dtype, Dynamic, Dynamic> >(0, 0, 0) {};
		void init(dtype *data, int rows, int cols) {
			new (this) Map<Matrix<dtype, Dynamic, Dynamic> >(data, rows, cols);
		};
};
