# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

cdef extern from "eigen_cpp.h":
    cdef cppclass VectorMap[Scalar]:
        VectorMap() nogil except +
        void init(Scalar *, int) nogil except +
        Scalar dot(const VectorMap[Scalar] &other) nogil except +

    cdef cppclass MatrixMap[Scalar]:
        MatrixMap() nogil
        void init(Scalar *, int, int) nogil except +


@cython.boundscheck(False)
cdef dtype dotvv(dtype[:] x, dtype[:] y) except *:
    cdef VectorMap[dtype] x_map, y_map
    x_map.init(&x[0], x.shape[0])
    y_map.init(&y[0], y.shape[0])
    return x_map.dot(y_map)

cdef dtype[:] dotmv(dtype[:, :] x, dtype[:] y) nogil:
    pass

cdef dtype[:] dotvm(dtype[:] x, dtype[:, :] y) nogil:
    pass

cdef dtype[:, :] dotmm(dtype[:, :] x, dtype[:, :] y) nogil:
    pass