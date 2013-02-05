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
        void noalias_assign_dot_mm "noalias_assign_dot_mm<NoContig, NoContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        # following are different template specializations, Ceygen unfortunately doesn't support yet function templates
        void noalias_assign_dot_cc "noalias_assign_dot_mm<CContig, CContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        void noalias_assign_dot_cf "noalias_assign_dot_mm<CContig, FContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        void noalias_assign_dot_fc "noalias_assign_dot_mm<FContig, CContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        void noalias_assign_dot_ff "noalias_assign_dot_mm<FContig, FContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +

        # exported Eigen methods
        Scalar determinant() nogil except +
        Scalar dot(BaseMap) nogil except +

        # this is a huge cheat, these operators don't map 1:1 to actual C++ operators at
        # all; but the declarations here are just to tell that the operators are possible..
        BaseMap operator+(BaseMap) nogil except +
        BaseMap operator-(BaseMap) nogil except +
        BaseMap operator*(BaseMap) nogil except +
        BaseMap operator/(BaseMap) nogil except +

    cdef cppclass VectorMap[Scalar](BaseMap):
        pass

    cdef cppclass RowVectorMap[Scalar](BaseMap):
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
