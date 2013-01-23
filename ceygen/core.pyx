# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython
from cython cimport view

cdef extern from "eigen_cpp.h":
    cdef cppclass VectorMap[Scalar]:
        VectorMap() nogil except +
        void init(Scalar *, const Py_ssize_t *) nogil except +
        Scalar dot(const VectorMap[Scalar] &other) nogil except +
        void noalias_assign(const VectorMap[Scalar] &other) nogil except +

    cdef cppclass MatrixMap[Scalar]:
        MatrixMap() nogil
        void init(Scalar *, const Py_ssize_t *) nogil except +
        VectorMap[Scalar] operator*(VectorMap[Scalar]) nogil except +

cdef str get_format(dtype *dummy):
    if dtype is double:
        return 'd'

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype dotvv(dtype[:] x, dtype[:] y) except *:
    cdef VectorMap[dtype] x_map, y_map
    x_map.init(&x[0], x.shape)
    y_map.init(&y[0], y.shape)
    return x_map.dot(y_map)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype[:] dotmv(dtype[:, :] x, dtype[:] y, dtype[:] out = None):
    cdef MatrixMap[dtype] x_map
    cdef VectorMap[dtype] y_map
    cdef VectorMap[dtype] out_map
    if out is None:
        out = view.array(shape=(x.shape[0],), itemsize=sizeof(dtype), format=get_format(&x[0, 0]))
    x_map.init(&x[0, 0], x.shape)
    y_map.init(&y[0], y.shape)
    out_map.init(&out[0], out.shape)
    out_map.noalias_assign(x_map * y_map)
    return out

cdef dtype[:] dotvm(dtype[:] x, dtype[:, :] y) nogil:
    pass

cdef dtype[:, :] dotmm(dtype[:, :] x, dtype[:, :] y) nogil:
    pass
