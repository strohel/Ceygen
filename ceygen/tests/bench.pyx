# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np
cdef object np_dot = np.dot
cdef object np_add = np.add
cdef object np_multiply = np.multiply

import os
import pickle
import subprocess
from time import time

from support import CeygenTestCase, benchmark
cimport ceygen.core as c
cimport ceygen.elemwise as e


class timeit:
    """Simple context manager to time interations of the block inside"""

    def __init__(self, func, implementation, args):
        self.func, self.iterations, self.cost = func, args['iterations'], args['cost']
        self.stats = args['self'].stats
        self.execute = True
        if implementation == 'numpy' and 'BENCHMARK_NUMPY' not in os.environ:
            self.execute = False
        self.implementation = implementation.rjust(args['self'].align)

    def __enter__(self):
        self.elapsed = time()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.elapsed = time() - self.elapsed
        if self.execute:
            percall = self.elapsed/self.iterations
            gflops = self.cost/percall/10.**9
            print "{0}: {1:.2e}s per call, {2:.3f}s total, {3:5.2f} GFLOPS".format(
                self.implementation, percall, self.elapsed, gflops)
            if self.implementation.endswith('ceygen') and isinstance(self.stats, dict):
                if self.func not in self.stats:
                    self.stats[self.func] = []
                self.stats[self.func].append(gflops)
        else:
            assert self.elapsed < 0.01
        return False  # let the exceptions fall through


class Bench(CeygenTestCase):

    align = 8

    def setUp(self):
        self.sizes = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
        if 'SAVE' in os.environ:
            self.stats = {}
        else:
            self.stats = None

    def tearDown(self):
        if self.stats:
            for (func, stats) in self.stats.iteritems():
                filename = func
                filename += b'-' + subprocess.check_output(['git', 'describe', '--dirty']).strip()
                filename += b'.pickle'
                with open(filename, 'wb') as f:
                    pickle.dump({'sizes': self.sizes, 'stats': stats}, f)
                    print "Saved stats to {0}".format(filename)

    @benchmark
    def test_bench_dot_mm(self):
        print __doc__
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

            with timeit(b"dot_mm", "numpy", locals()) as context:
                if context.execute:
                    for i in range(context.iterations):
                        np_dot(x_np, y_np, out_np)
            with timeit(b"dot_mm", "ceygen", locals()) as context:
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

            with timeit(b"add_vv", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_add(x_np, x_np, out_np)
            with timeit(b"add_vv", "ceygen", locals()) as context:
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

            with timeit(b"multiply_mm", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_multiply(x_np, x_np, out_np)
            with timeit(b"multiply_mm", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        e.multiply_mm(x, x, out)
