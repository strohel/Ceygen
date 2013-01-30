# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cdef extern from "eigen_cpp.h":
    cdef cppclass VectorMap[Scalar]:
        VectorMap() nogil except +
        void init(Scalar *, const Py_ssize_t *, const Py_ssize_t *) nogil except +
        # note: Cython doesn't like VectorMap[Scalar] anywhere except in cdef cppclass ...
        # declaration. Using just the class without template param everywhere else works
        # well
        VectorMap transpose() nogil  # should never raise an Exception

        Scalar dot(VectorMap) nogil except +
        VectorMap operator+(VectorMap) nogil except +
        VectorMap operator-(VectorMap) nogil except +
        VectorMap operator*(MatrixMap) nogil except +

        void noalias_assign(VectorMap) nogil except +

    cdef cppclass MatrixMap[Scalar]:
        MatrixMap() nogil
        void init(Scalar *, const Py_ssize_t *, const Py_ssize_t *) nogil except +
        MatrixMap transpose() nogil

        MatrixMap operator+(MatrixMap) nogil except +
        MatrixMap operator-(MatrixMap) nogil except +
        VectorMap operator*(VectorMap) nogil except +
        MatrixMap operator*(MatrixMap) nogil except +

        void noalias_assign(MatrixMap) nogil except +
        void assign_inverse(MatrixMap) nogil except +
