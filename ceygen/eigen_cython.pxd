# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# note: Cython doesn't like VectorMap[Scalar] anywhere except in cdef cppclass ...
# declaration. Using just the class without template param everywhere else works
# well

cdef extern from "eigen_cpp.h":
    cdef cppclass BaseMap[Scalar]:
        # "constructor":
        void init(Scalar *, const Py_ssize_t *, const Py_ssize_t *) nogil except +

        # our own methods:
        void assign(BaseMap) nogil except +
        void assign_inverse(BaseMap) nogil except +
        void noalias_assign(BaseMap) nogil except +

        # exported Eigen methods
        Scalar determinant() nogil except +
        Scalar dot(BaseMap) nogil except +
        BaseMap transpose() nogil  # should never raise an Exception

        # this is a huge cheat, these operators don't map 1:1 to actual C++ operators at
        # all; but the declarations here are just to tell that the operators are possible..
        BaseMap operator+(BaseMap) nogil except +
        BaseMap operator-(BaseMap) nogil except +
        BaseMap operator*(BaseMap) nogil except +
        BaseMap operator/(BaseMap) nogil except +

    cdef cppclass VectorMap[Scalar](BaseMap):
        pass

    cdef cppclass Array1DMap[Scalar](BaseMap):
        # must be here, Cython has problems inheriting overloads, http://trac.cython.org/cython_trac/ticket/800
        BaseMap operator+(Scalar) nogil except +
        BaseMap operator*(Scalar) nogil except +

    cdef cppclass MatrixMap[Scalar](BaseMap):
        pass

    cdef cppclass Array2DMap[Scalar](BaseMap):
        # http://trac.cython.org/cython_trac/ticket/800
        BaseMap operator+(Scalar) nogil except +
        BaseMap operator*(Scalar) nogil except +
