# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np
np_dot = np.dot

from time import time

from support import CeygenTestCase
cimport ceygen.core as c


class Bench(CeygenTestCase):

    def test_dot(self):
        print
        cdef int iterations
        cdef double[:, :] x
        cdef double[:, :] y
        cdef double[:, :] out

        for size in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100):
            x_np = np.random.rand(size, size)
            y_np = np.random.rand(size, size)
            out_np = np.empty((size, size))
            x = x_np
            y = y_np
            out = out_np

            cost = size**3.
            iterations = min(3.*10.**9. / cost, 1000000)
            print "size: {0}x{0}, iterations: {1}".format(size, iterations)

            elapsed = time()
            for i in range(iterations):
                np_dot(x_np, y_np, out_np)
            elapsed = time() - elapsed
            print "       numpy: {0}s per call, {1}s total".format(elapsed/iterations, elapsed)
            out_copy = out.copy()

            elapsed = time()
            for i in range(iterations):
                c.dot_mm(x, y, out)
            elapsed = time() - elapsed
            print "      ceygen: {0}s per call, {1}s total".format(elapsed/iterations, elapsed)

            self.assertApproxEqual(out, out_copy)
