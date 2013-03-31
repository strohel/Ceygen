/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

#ifndef EIGEN_CPP_H
#define EIGEN_CPP_H
// two macros ensures any macro passed will be expanded before being stringified
#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)

#include <stdexcept>
// make Eigen raise an exception instead of aborting on assert failure. Cython converts
// std::runtime_error to Python RuntimeError
#define eigen_assert(statement) do { if(!(statement)) throw std::invalid_argument(#statement " does not hold in " __FILE__ ":" STRINGIZE(__LINE__)); } while(0)
#define EIGEN_NO_AUTOMATIC_RESIZING // affects operator=, Ceygen doesn't want resizing
#define EIGEN_RUNTIME_NO_MALLOC // enables use of set_is_malloc_allowed() in tests

#include <Eigen/Core>
#include <Eigen/LU> // for Matrix.inverse()
#include <Eigen/Cholesky> // for Matrix.llt()

#include <Python.h> // for Py_ssize_t

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

		BaseMap() : Base(0,
				BaseType::RowsAtCompileTime == Dynamic ? 0 : BaseType::RowsAtCompileTime,
				BaseType::ColsAtCompileTime == Dynamic ? 0 : BaseType::ColsAtCompileTime,
				StrideType(
						StrideType::OuterStrideAtCompileTime == Dynamic ? 0 : StrideType::OuterStrideAtCompileTime,
						StrideType::InnerStrideAtCompileTime == Dynamic ? 0 : StrideType::InnerStrideAtCompileTime
				)
		) {}

		inline void init(Scalar *data, const Py_ssize_t *shape, const Py_ssize_t *strides) {
			// enum is used just to ensure that this is a compile-time constant
			enum {
				RowsShapeIndex = 0, // for both vectors and matrices; entry exists just for symmetry
				ColsShapeIndex = Base::IsVectorAtCompileTime ? 0 : 1,
				OuterStrideIndex = (BaseType::Options & RowMajor) ? 0 : 1, // only used for matrices
				InnerStrideIndex = Base::IsVectorAtCompileTime ? 0 : ((BaseType::Options & RowMajor) ? 1 : 0),
			};

#			ifdef DEBUG
				std::cerr << __PRETTY_FUNCTION__ << std::endl;
				std::cerr << "got: shape: " << shape[0] << ", " << shape[1] << " strides: " << strides[0] << ", " << strides[1] << std::endl;
				std::cerr << "got: RowsAtCompileTime: " << BaseType::RowsAtCompileTime << " ColsAtCompileTime: " << BaseType::ColsAtCompileTime << " Options: " << BaseType::Options << std::endl;
				std::cerr << "got: OuterStrideAtCompileTime: " << StrideType::OuterStrideAtCompileTime << " InnerStrideAtCompileTime: " << StrideType::InnerStrideAtCompileTime << std::endl;
				std::cerr << "got: RowsShapeIndex: " << RowsShapeIndex << " ColsShapeIndex: " << ColsShapeIndex << " OuterStrideIndex: " << OuterStrideIndex << " InnerStrideIndex: " << InnerStrideIndex << std::endl;
#			endif

			/* see http://eigen.tuxfamily.org/dox/TutorialMapClass.html - this is NOT a heap allocation
			 * Note: Cython (and Python) has strides in bytes, Eigen in sizeof(Scalar) units */
			new (this) Base(data,
					BaseType::RowsAtCompileTime == Dynamic ? shape[RowsShapeIndex] : BaseType::RowsAtCompileTime,
					BaseType::ColsAtCompileTime == Dynamic ? shape[ColsShapeIndex] : BaseType::ColsAtCompileTime,
					StrideType(
							Base::IsVectorAtCompileTime ? 0 : strides[OuterStrideIndex]/sizeof(Scalar),
							strides[InnerStrideIndex]/sizeof(Scalar)
					)
			);
#			ifdef DEBUG
				std::cerr << "rows=" << this->rows() << ", cols=" << this->cols()
				          << " outerStride=" << this->outerStride() << ", innerStride=" << this->innerStride() << std::endl;
				bool malloc_allowed = internal::is_malloc_allowed();
				internal::set_is_malloc_allowed(true);
				std::cerr << *this << std::endl;
				internal::set_is_malloc_allowed(malloc_allowed);
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

		EIGEN_INHERIT_ASSIGNMENT_OPERATORS(BaseMap)
};

template<typename dtype, typename ContiguityType>
class VectorMap : public BaseMap<Matrix<dtype, Dynamic, 1>, Stride<0, ContiguityType::InnerStride> >
{
};

template<typename dtype, typename ContiguityType>
class RowVectorMap : public BaseMap<Matrix<dtype, 1, Dynamic>, Stride<0, ContiguityType::InnerStride> >
{
};

template<typename dtype, typename ContiguityType>
class Array1DMap : public BaseMap<Array<dtype, Dynamic, 1>, Stride<0, ContiguityType::InnerStride> >
{
};

template<typename dtype, typename ContiguityType>
class MatrixMap : public BaseMap<Matrix<dtype, Dynamic, Dynamic, ContiguityType::Layout>, Stride<Dynamic, ContiguityType::InnerStride> >
{
};

template<typename dtype, typename ContiguityType>
class Array2DMap : public BaseMap<Array<dtype, Dynamic, Dynamic, ContiguityType::Layout>, Stride<Dynamic, ContiguityType::InnerStride> >
{
};

#endif // EIGEN_CPP_H
