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
template<typename BaseType, typename StrideType>
class BaseMap : public Map<BaseType, Unaligned, StrideType>
{
	public:
		typedef Map<BaseType, Unaligned, StrideType> Base;
		typedef typename Base::Scalar Scalar;

		BaseMap() : Base(0, 0, BaseType::ColsAtCompileTime == Dynamic ? 0 : BaseType::ColsAtCompileTime, StrideType(0, 0)) {}

		inline void init(Scalar *data, const Py_ssize_t *shape, const Py_ssize_t *strides) {
			// see http://eigen.tuxfamily.org/dox/TutorialMapClass.html
			// this is NOT a heap allocation
			// Cython has strides in bytes, Eigen in Scalar-long units:
			new (this) Base(data, shape[0],
					BaseType::ColsAtCompileTime == Dynamic ? shape[1] : BaseType::ColsAtCompileTime,
					StrideType(
							StrideType::OuterStrideAtCompileTime == Dynamic ? strides[0]/sizeof(Scalar) : StrideType::OuterStrideAtCompileTime,
							StrideType::OuterStrideAtCompileTime == Dynamic ? strides[1]/sizeof(Scalar) : strides[0]/sizeof(Scalar)));
#			ifdef DEBUG
				std::cerr << __PRETTY_FUNCTION__ << " rows=" << this->rows() << ", cols=" << this->cols()
				          << " outerStride=" << this->outerStride() << ", innerStride=" << this->innerStride() << std::endl
				          << *this << std::endl;
#			endif
		};

		// if we write "x = y" in a .pyx file, Cython creates a temporary, which breaks
		// Matrix's operator= and needlessly copies memory
		template<typename T>
		inline void assign(const T &rhs) {
			*this = rhs;
		}

		// see above
		template<typename T>
		inline void assign_inverse(const T &rhs) {
			*this = rhs.inverse();
		}

		template<typename T>
		inline void noalias_assign(const T &rhs) {
			this->noalias() = rhs;
		}

		EIGEN_INHERIT_ASSIGNMENT_OPERATORS(BaseMap)
};

template<typename dtype>
class VectorMap : public BaseMap<Matrix<dtype, Dynamic, 1>, Stride<0, Dynamic> >
{
};

template<typename dtype>
class Array1DMap : public BaseMap<Array<dtype, Dynamic, 1>, Stride<0, Dynamic> >
{
};

template<typename dtype>
class MatrixMap : public BaseMap<Matrix<dtype, Dynamic, Dynamic, RowMajor>, Stride<Dynamic, Dynamic> >
{
};

template<typename dtype>
class Array2DMap : public BaseMap<Array<dtype, Dynamic, Dynamic, RowMajor>, Stride<Dynamic, Dynamic> >
{
};
