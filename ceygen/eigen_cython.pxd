# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# note: Cython doesn't like VectorMap[Scalar] anywhere except in cdef cppclass ...
# declaration. Using just the class without template param everywhere else works
# well

from libcpp cimport bool

from dtype cimport dtype


cdef extern from "eigen_cpp.h":
    void c_set_is_malloc_allowed "internal::set_is_malloc_allowed"(bool) nogil
    int EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION

    cdef cppclass BaseMap[Scalar]:
        # "constructor":
        void init(Scalar *, const Py_ssize_t *, const Py_ssize_t *) nogil

        # our own methods:
        void assign(BaseMap) nogil
        void assign_inverse(BaseMap) nogil
        void noalias_assign(BaseMap) nogil

        # exported Eigen methods
        Scalar determinant() nogil
        Scalar dot(BaseMap) nogil
        # a little hack so that we don't have to introduce VectorwiseOp cppclass:
        BaseMap colwise_sum "colwise().sum"() nogil
        BaseMap rowwise_sum "rowwise().sum"() nogil
        # a little hack so that we don't have to introduce LLT class:
        BaseMap llt_matrixL "llt().matrixL"() nogil
        Scalar sum() nogil
        BaseMap pow(Scalar) nogil

        # this is a huge cheat, these operators don't map 1:1 to actual C++ operators at
        # all; but the declarations here are just to tell that the operators are possible..
        BaseMap operator+(BaseMap) nogil
        BaseMap operator-(BaseMap) nogil
        BaseMap operator*(BaseMap) nogil
        BaseMap operator/(BaseMap) nogil

    cdef cppclass VectorMap[Scalar, ContiguityType](BaseMap):
        pass

    cdef cppclass RowVectorMap[Scalar, ContiguityType](BaseMap):
        pass

    cdef cppclass Array1DMap[Scalar, ContiguityType](BaseMap):
        # must be here, Cython has problems inheriting overloads, http://trac.cython.org/cython_trac/ticket/800
        BaseMap operator+(Scalar) nogil
        BaseMap operator*(Scalar) nogil

    cdef cppclass MatrixMap[Scalar, ContiguityType](BaseMap):
        pass

    cdef cppclass Array2DMap[Scalar, ContiguityType](BaseMap):
        # http://trac.cython.org/cython_trac/ticket/800
        BaseMap operator+(Scalar) nogil
        BaseMap operator*(Scalar) nogil
