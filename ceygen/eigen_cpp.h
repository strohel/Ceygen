/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

#include <stdexcept>
// make Eigen raise an exception instead of aborting on assert failure. Cython converts
// std::runtime_error to Python RuntimeError
#define eigen_assert(statement) do { if(!(statement)) throw std::runtime_error(#statement " does not hold (in Eigen)"); } while(0)

#include <Eigen/Core>
#include <Python.h>

using namespace Eigen;

/**
 * Very simple Eigen::Map<> subclass that provides default constructor and lets
 * Cython late-initialize the map using init() method
 */
template<typename dtype>
class VectorMap : public Map<Matrix<dtype, Dynamic, 1> >
{
	public:
		typedef Map<Matrix<dtype, Dynamic, 1> > Base;

		VectorMap() : Base(0, 0) {}
		inline void init(dtype *data, Py_ssize_t *shape) {
			// see http://eigen.tuxfamily.org/dox/TutorialMapClass.html
			// this is NOT a heap allocation:
			new (this) Base(data, shape[0]);
		};

		template<typename T>
		inline void noalias_assign(const T &rhs) {
			this->noalias() = rhs;
		}

		EIGEN_INHERIT_ASSIGNMENT_OPERATORS(VectorMap)
};

/**
 * @see VectorMap
 */
template<typename dtype>
class MatrixMap : public Map<Matrix<dtype, Dynamic, Dynamic, RowMajor> >
{
	public:
		typedef Map<Matrix<dtype, Dynamic, Dynamic, RowMajor> > Base;

		MatrixMap() : Base(0, 0, 0) {};
		inline void init(dtype *data, const Py_ssize_t *shape) {
			new (this) Base(data, shape[0], shape[1]);
		};

		EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixMap)
};
