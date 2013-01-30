/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

#include <stdexcept>
// make Eigen raise an exception instead of aborting on assert failure. Cython converts
// std::runtime_error to Python RuntimeError
#define eigen_assert(statement) do { if(!(statement)) throw std::invalid_argument(#statement " does not hold (in Eigen)"); } while(0)

#include <Eigen/Core>
#include <Eigen/LU> // for Matrix.inverse()

#include <Python.h>

#ifdef DEBUG
#include <iostream>
#endif

using namespace Eigen;

/**
 * Very simple Eigen::Map<> subclass that provides default constructor and lets
 * Cython late-initialize the map using init() method
 */
template<typename dtype>
class VectorMap : public Map<Matrix<dtype, Dynamic, 1>, Unaligned, Stride<0, Dynamic> >
{
	public:
		typedef Stride<0, Dynamic> StrideType;
		typedef Map<Matrix<dtype, Dynamic, 1>, Unaligned, StrideType> Base;

		VectorMap() : Base(0, 0, StrideType(0, 0)) {}
		inline void init(dtype *data, const Py_ssize_t *shape, const Py_ssize_t *strides) {
			// see http://eigen.tuxfamily.org/dox/TutorialMapClass.html
			// this is NOT a heap allocation
			// Cython has strides in bytes, Eigen in dtype-long units:
			new (this) Base(data, shape[0], StrideType(0, strides[0]/sizeof(dtype)));
#			ifdef DEBUG
				std::cerr << __PRETTY_FUNCTION__ << " shape=" << shape[0]
				          << " strides=" << strides[0]/sizeof(dtype) << std::endl
				          << *this << std::endl;
#			endif
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
class MatrixMap : public Map<Matrix<dtype, Dynamic, Dynamic, RowMajor>, Unaligned, Stride<Dynamic, Dynamic> >
{
	public:
		typedef Stride<Dynamic, Dynamic> StrideType;
		typedef Map<Matrix<dtype, Dynamic, Dynamic, RowMajor>, Unaligned, StrideType> Base;

		MatrixMap() : Base(0, 0, 0, StrideType(0, 0)) {};
		inline void init(dtype *data, const Py_ssize_t *shape, const Py_ssize_t *strides) {
			new (this) Base(data, shape[0], shape[1], StrideType(strides[0]/sizeof(dtype), strides[1]/sizeof(dtype)));
#			ifdef DEBUG
				std::cerr << __PRETTY_FUNCTION__ << " shape=" << shape[0] << ", " << shape[1]
				          << " strides=" << strides[0]/sizeof(dtype) << ", " << strides[1]/sizeof(dtype) << std::endl
				          << *this << std::endl;
#			endif
		};

		template<typename T>
		inline void noalias_assign(const T &rhs) {
			this->noalias() = rhs;
		}

		// this is a HACK because if we write "x = y.inverse()" in a .pyx file, Cython
		// creates a temporary, which breaks Matrix's operator= and needless copies memory
		template<typename T>
		inline void assign_inverse(const T &rhs) {
			*this = rhs.inverse();
		}

		EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixMap)
};
