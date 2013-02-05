# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np
np_dot = np.dot

from time import time

from support import CeygenTestCase, benchmark
cimport ceygen.core as c


class timeit:
    """Simple context manager to time interations of the block inside"""

    def __init__(self, name, align, iterations, cost):
        self.iterations, self.cost = iterations, cost
        self.name = name.rjust(align)

    def __enter__(self):
        self.elapsed = time()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.elapsed = time() - self.elapsed
        percall = self.elapsed/self.iterations
        print "{0}: {1:.2e}s per call, {2:.3f}s total, {3:5.2f} GFLOPS".format(self.name,
              percall, self.elapsed, self.cost/percall/10.**9)
        return False  # let the exceptions fall through


class Bench(CeygenTestCase):

    @benchmark
    def test_bench_dot_mm(self):
        print
        cdef int iterations
        cdef double[:, :] x, x_nocontig, y, out

        for size in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100):
            x_np = np.random.rand(size, size)
            #x_nocontig_np = np.random.rand(size, size, 2)[:, :, 0]
            y_np = np.random.rand(size, size)
            out_np = np.empty((size, size))
            x = x_np
            #x_nocontig = x_nocontig_np
            y = y_np
            out = out_np

            cost = 2. * size**3.
            iterations = min(2. * 10.**9. / cost, 1000000)
            print "size: {0}x{0}, iterations: {1}".format(size, iterations)
            align = 8

            with timeit("numpy", align, iterations, cost):
                for i in range(iterations):
                    np_dot(x_np, y_np, out_np)
            with timeit("ceygen", align, iterations, cost):
                for i in range(iterations):
                    c.dot_mm(x, y, out)

            #iterations /= 3
            #with timeit("numpy-nocontig", align, iterations, cost):
                #for i in range(iterations):
                    #np_dot(x_nocontig_np, y_np, out_np)
            #with timeit("ceygen-nocontig", align, iterations, cost):
                #for i in range(iterations):
                    #c.dot_mm(x_nocontig, y, out)
