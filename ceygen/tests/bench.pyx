# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np
cdef object np_dot = np.dot
cdef object np_add = np.add
cdef object np_multiply = np.multiply

import os
from time import time

from support import CeygenTestCase, benchmark
cimport ceygen.core as c
cimport ceygen.elemwise as e


class timeit:
    """Simple context manager to time interations of the block inside"""

    def __init__(self, name, align, iterations, cost):
        self.iterations, self.cost = iterations, cost
        self.execute = True
        if name == 'numpy' and 'BENCHMARK_NUMPY' not in os.environ:
            self.execute = False
        self.name = name.rjust(align)

    def __enter__(self):
        self.elapsed = time()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.elapsed = time() - self.elapsed
        if self.execute:
            percall = self.elapsed/self.iterations
            print "{0}: {1:.2e}s per call, {2:.3f}s total, {3:5.2f} GFLOPS".format(self.name,
                percall, self.elapsed, self.cost/percall/10.**9)
        else:
            assert self.elapsed < 0.01
        return False  # let the exceptions fall through


class Bench(CeygenTestCase):

    sizes = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    align = 8

    @benchmark
    def test_bench_dot_mm(self):
        print
        cdef int iterations
        cdef double[:, :] x, y, out
        for size in self.sizes:
            x_np = np.random.rand(size, size)
            y_np = np.random.rand(size, size)
            out_np = np.empty((size, size))
            x, y, out = x_np, y_np, out_np

            cost = 2. * size**3.
            iterations = min(max(2. * 10.**9. / cost, 1), 1000000)
            print "size: {0}*{0}, iterations: {1}".format(size, iterations)

            with timeit("numpy", self.align, iterations, cost) as context:
                if context.execute:
                    for i in range(iterations):
                        np_dot(x_np, y_np, out_np)
            with timeit("ceygen", self.align, iterations, cost) as context:
                if context.execute:
                    for i in range(iterations):
                        c.dot_mm(x, y, out)

    @benchmark
    def test_bench_add_vv(self):
        print
        cdef int iterations
        cdef double[:] x, out

        for size in self.sizes:
            x_np = np.random.rand(size)
            out_np = np.empty(size)
            x, out = x_np, out_np

            cost = size
            iterations = min(0.25 * 10.**9 / cost, 1000000)
            print "size: {0}, iterations: {1}".format(size, iterations)

            with timeit("numpy", self.align, iterations, cost) as context:
                if context.execute:
                    for i in range(iterations):
                        np_add(x_np, x_np, out_np)
            with timeit("ceygen", self.align, iterations, cost) as context:
                if context.execute:
                    for i in range(iterations):
                        e.add_vv(x, x, out)

    @benchmark
    def test_bench_multiply_mm(self):
        print
        cdef int iterations
        cdef double[:, :] x, out

        for size in self.sizes:
            x_np = np.random.rand(size, size)
            out_np = np.empty((size, size))
            x, out = x_np, out_np

            cost = size**2.
            iterations = min(0.25 * 10.**9 / cost, 1000000)
            print "size: {0}*{0}, iterations: {1}".format(size, iterations)

            with timeit("numpy", self.align, iterations, cost) as context:
                if context.execute:
                    for i in range(iterations):
                        np_multiply(x_np, x_np, out_np)
            with timeit("ceygen", self.align, iterations, cost) as context:
                if context.execute:
                    for i in range(iterations):
                        e.multiply_mm(x, x, out)
