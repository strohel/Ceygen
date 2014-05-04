# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from cython.parallel cimport prange

import numpy as np
cdef object np_dot = np.dot
cdef object np_add = np.add
cdef object np_multiply = np.multiply
cdef object np_det = np.linalg.det

import os
import pickle
import subprocess
from time import time

from support import CeygenTestCase, benchmark
cimport ceygen.core as c
cimport ceygen.elemwise as e
cimport ceygen.lu as lu


class timeit:
    """Simple context manager to time interations of the block inside"""

    def __init__(self, func, implementation, args):
        self.func, self.iterations, self.cost = func, args['iterations'], args['cost']
        self.stats = args['self'].stats
        self.percall = args['self'].percall
        self.execute = True
        if implementation.startswith('numpy') and 'BENCHMARK_NUMPY' not in os.environ:
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
            if isinstance(self.stats, dict):
                key = self.func + '.' + self.implementation.strip()
                if key not in self.stats:
                    self.stats[key] = []
                    self.percall[key] = []
                self.stats[key].append(gflops)
                self.percall[key].append(percall)
        else:
            assert self.elapsed < 0.01
        return False  # let the exceptions fall through


class Bench(CeygenTestCase):

    align = 8

    def setUp(self):
        self.sizes = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
        if 'SAVE' in os.environ:
            self.stats = {}
            self.percall = {}
        else:
            self.stats = None
            self.percall = None

    def tearDown(self):
        if self.stats:
            for (func, stats) in self.stats.iteritems():
                percall = self.percall[func]
                filename = func
                filename += b'-' + subprocess.check_output(['git', 'describe', '--dirty']).strip()
                filename += b'.pickle'
                with open(filename, 'wb') as f:
                    pickle.dump({'sizes': self.sizes, 'stats': stats, 'percall': percall}, f)
                    print "Saved stats to {0}".format(filename)

    # core module

    @benchmark
    def test_bench_dot_vv(self):
        print
        cdef int iterations
        cdef double[:] x

        for size in self.sizes:
            x_np = np.random.rand(size)
            x = x_np

            cost = 2. * size
            iterations = min(1.0 * 10.**9 / cost, 1000000)
            print "size: {0}, iterations: {1}".format(size, iterations)

            with timeit(b"dot_vv", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        c.dot_vv(x, x)

    @benchmark
    def test_bench_dot_mv(self):
        print
        cdef int iterations
        cdef double[:, :] x
        cdef double[:] y, out

        for size in self.sizes:
            x_np = np.random.rand(size, size)
            y_np = np.random.rand(size)
            out_np = np.empty(size)
            x, y, out = x_np, y_np, out_np

            cost = 2. * size**2.
            iterations = min(0.5 * 10.**9. / cost, 1000000)
            print "size: {0}, iterations: {1}".format(size, iterations)

            with timeit(b"dot_mv", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_dot(x_np, y_np, out_np)
            with timeit(b"dot_mv", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        c.dot_mv(x, y, out)

    @benchmark
    def test_bench_dot_mv_noout(self):
        print
        cdef int iterations
        cdef double[:, :] x
        cdef double[:] y

        for size in self.sizes:
            x_np = np.random.rand(size, size)
            y_np = np.random.rand(size)
            x, y = x_np, y_np

            cost = 2. * size**2.
            iterations = min(0.5 * 10.**9. / cost, 1000000)
            print "size: {0}, iterations: {1}".format(size, iterations)

            with timeit(b"dot_mv_noout", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_dot(x_np, y_np)
            with timeit(b"dot_mv_noout", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        c.dot_mv(x, y)

    @benchmark
    def test_bench_dot_vm(self):
        print
        cdef int iterations
        cdef double[:] x, out
        cdef double[:, :] y

        for size in self.sizes:
            x_np = np.random.rand(size)
            y_np = np.random.rand(size, size)
            out_np = np.empty(size)
            x, y, out = x_np, y_np, out_np

            cost = 2. * size**2.
            iterations = min(0.5 * 10.**9. / cost, 1000000)
            print "size: {0}, iterations: {1}".format(size, iterations)

            with timeit(b"dot_vm", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_dot(x_np, y_np, out_np)
            with timeit(b"dot_vm", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        c.dot_vm(x, y, out)

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

            with timeit(b"dot_mm", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_dot(x_np, y_np, out_np)
            with timeit(b"dot_mm", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        c.dot_mm(x, y, out)

    # elemwise module

    @benchmark
    def test_bench_multiply_vs(self):
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

            with timeit(b"multiply_vs", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        e.multiply_vs(x, 12., out)

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
    def test_bench_add_ms(self):
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

            with timeit(b"add_ms", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        e.add_ms(x, -11., out)

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

    # lu module

    @benchmark
    def test_bench_inv(self):
        print
        cdef int iterations
        cdef double[:, :] x, out

        for size in self.sizes:
            x_np = np.random.rand(size, size)
            out_np = np.empty((size, size))
            x, out = x_np, out_np

            cost = size**3.  # 2/3 * n^3 floating point operations, but additional logic operations
            iterations = min(max(0.25 * 10.**9. / cost, 1), 1000000)
            print "size: {0}*{0}, iterations: {1}".format(size, iterations)

            with timeit(b"inv", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        lu.inv(x, out)

    @benchmark
    def test_bench_iinv(self):
        print
        cdef int iterations
        cdef double[:, :] x

        for size in self.sizes:
            x_np = np.random.rand(size, size)
            x = x_np

            cost = size**3.  # 2/3 * n^3 floating point operations, but additional logic operations
            iterations = min(max(0.25 * 10.**9. / cost, 1), 1000000)
            print "size: {0}*{0}, iterations: {1}".format(size, iterations)

            with timeit(b"iinv", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        lu.iinv(x)

    @benchmark
    def test_bench_det(self):
        print
        cdef int i, iterations
        cdef double[:, :] x
        from multiprocessing import cpu_count, Pool

        for size in self.sizes:
            x_np = np.random.rand(size, size)
            x = x_np

            cost = size**3.
            iterations = 4 * min(max(int(0.25 * 10.**9. / cost), 1), 250000)
            print "size: {0}*{0}, iterations: {1}".format(size, iterations)
            origalign = self.align
            self.align = 17

            with timeit(b"det", "numpy", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        np_det(x_np)

            with timeit(b"det", "ceygen", locals()) as context:
                if context.execute:
                    for i in range(iterations):
                        lu.det(x)

            with timeit(b"det", "numpy parallel", locals()) as context:
                if context.execute:
                    pool = Pool(processes=cpu_count())  # TODO: actual number of cores
                    pool.map(np_det, (x_np for i in range(iterations)))

            with timeit(b"det", "ceygen parallel", locals()) as context:
                if context.execute:
                    for i in prange(iterations, nogil=True):
                        lu.det(x)

            self.align = origalign
