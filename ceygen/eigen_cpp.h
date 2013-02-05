/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

// two macros ensures any macro passed will be expanded before being stringified
#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)

#include <stdexcept>
// make Eigen raise an exception instead of aborting on assert failure. Cython converts
// std::runtime_error to Python RuntimeError
#define eigen_assert(statement) do { if(!(statement)) throw std::invalid_argument(#statement " does not hold in " __FILE__ ":" STRINGIZE(__LINE__)); } while(0)
#define EIGEN_NO_AUTOMATIC_RESIZING // affects operator=, Ceygen doesn't want resizing

#include <Eigen/Core>
#include <Eigen/LU> // for Matrix.inverse()

#include <Python.h>

#ifdef DEBUG
#include <iostream>
#endif

using namespace Eigen;

// for noalias_assign_dot_mm():
enum Contiguity {
	NoContig,
	CConting,
	FContig
};

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

		BaseMap() : Base(0,
				BaseType::RowsAtCompileTime == Dynamic ? 0 : BaseType::RowsAtCompileTime,
				BaseType::ColsAtCompileTime == Dynamic ? 0 : BaseType::ColsAtCompileTime,
				StrideType(0, 0)) {}

		inline void init(Scalar *data, const Py_ssize_t *shape, const Py_ssize_t *strides) {
			/* which index inside shape, strides to use as "column" (unless known at compile
			 * time) and "innerStride" for Eigen? Depends on whether this is a matrix (then
			 * it should be 1) or a vector, where this should be 0. enum is used just to
			 * ensure that this is a compile-time constant */
			enum { ColsShapeIndex = Base::IsVectorAtCompileTime ? 0 : 1 };
			/* see http://eigen.tuxfamily.org/dox/TutorialMapClass.html - this is NOT a heap allocation
			 * Note: Cython (and Python) has strides in bytes, Eigen in sizeof(Scalar) units */
			new (this) Base(data,
					BaseType::RowsAtCompileTime == Dynamic ? shape[0] : BaseType::RowsAtCompileTime,
					BaseType::ColsAtCompileTime == Dynamic ? shape[ColsShapeIndex] : BaseType::ColsAtCompileTime,
					StrideType(
							StrideType::OuterStrideAtCompileTime == Dynamic ? strides[0]/sizeof(Scalar) : StrideType::OuterStrideAtCompileTime,
							strides[ColsShapeIndex]/sizeof(Scalar)));
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
			assign(rhs.inverse());
		}

		template<typename T>
		inline void noalias_assign(const T &rhs) {
			this->noalias() = rhs;
		}

		/** Very special optimization for dot_mm: TODO
		 */
		template<Contiguity XContiguity, Contiguity YContiguity>
		inline void noalias_assign_dot_mm(
				Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
				Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides)
		{
			typedef Stride<Dynamic, Dynamic> MyStride;
			typedef Map<Matrix<Scalar, Dynamic, Dynamic, RowMajor>, Unaligned, MyStride> MyMatrix;
			MyMatrix x(x_data, x_shape[0], x_shape[1], MyStride(x_strides[0]/sizeof(Scalar), x_strides[1]/sizeof(Scalar)));
			MyMatrix y(y_data, y_shape[0], y_shape[1], MyStride(y_strides[0]/sizeof(Scalar), y_strides[1]/sizeof(Scalar)));
			noalias_assign(x * y);
#			ifdef DEBUG
				std::cerr << __PRETTY_FUNCTION__ << " x:" << std::endl << x << std::endl;
				std::cerr << __PRETTY_FUNCTION__ << " y:" << std::endl << y << std::endl;
#			endif
		}

		EIGEN_INHERIT_ASSIGNMENT_OPERATORS(BaseMap)
};

template<typename dtype>
class VectorMap : public BaseMap<Matrix<dtype, Dynamic, 1>, Stride<0, Dynamic> >
{
};

template<typename dtype>
class RowVectorMap : public BaseMap<Matrix<dtype, 1, Dynamic>, Stride<0, Dynamic> >
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
