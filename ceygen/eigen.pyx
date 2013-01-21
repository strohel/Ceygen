# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cdef extern from "eigen.h":
    cdef cppclass VectorMap[Scalar]:
        VectorMap() nogil
        void init(Scalar *, int) nogil
        Scalar dot(const VectorMap[Scalar] &other) nogil

    cdef cppclass MatrixMap[Scalar]:
        MatrixMap() nogil
        void init(Scalar *, int, int) nogil


cdef dtype dotvv(dtype[:] x, dtype[:] y) nogil:
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
