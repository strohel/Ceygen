# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np

from support import NoMallocTestCase
cimport ceygen.reductions as r


class TestReductions(NoMallocTestCase):

    def test_sum_v(self):
        x_np = np.array([1., 3., 5.])
        expected = 9.

        self.assertApproxEqual(r.sum_v[double](x_np), expected)
        cdef double[:] x = x_np
        self.assertApproxEqual(r.sum_v(x), expected)
        # sum of None (which can be interpreted as an empty vector) is defined to be 0.0
        self.assertApproxEqual(r.sum_v[double](None), 0.)

    def test_sum_v_badargs(self):
        with self.assertRaises(ValueError):
            r.sum_v[double](np.array([[1.]]))
        with self.assertRaises(TypeError):
            l = [1, 2, 3]
            r.sum_v[double](l)

    def test_sum_m(self):
        x_np = np.array([[1., 3., 5.]])
        expected = 9.

        self.assertApproxEqual(r.sum_m[double](x_np), expected)
        cdef double[:, :] x = x_np
        self.assertApproxEqual(r.sum_m(x), expected)
        # sum of None (which can be interpreted as an empty matrix) is defined to be 0.0
        self.assertApproxEqual(r.sum_m[double](None), 0.)

    def test_sum_m_badargs(self):
        with self.assertRaises(ValueError):
            r.sum_m[double](np.array([1.]))
        with self.assertRaises(TypeError):
            l = [[1, 2, 3]]
            r.sum_m[double](l)

    def test_rowwise_sum(self):
        x_np = np.array([[2., 4., 6.],
                         [1., 3., 5.]])
        expected, expected_t = [12., 9.], [3., 7., 11.]

        self.assertApproxEqual(r.rowwise_sum[double](x_np), expected)
        self.assertApproxEqual(r.rowwise_sum[double](x_np.T), expected_t)
        out_np = np.empty(2)
        out2 = r.rowwise_sum[double](x_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:, :] x = x_np
        self.assertApproxEqual(r.rowwise_sum(x), expected)
        out_np[:] = -123.  # reset so that we would catch errors
        cdef double[:] out = out_np
        out2 = r.rowwise_sum(x, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_rowwise_sum_badargs(self):
        x = np.array([[1., 2., 3.]])
        out = np.empty(1)

        for X in (x, np.array([1., 2.]), np.array([[1.], [2.]]), None):
            for OUT in (out, np.array([[1.]])):
                if X is x and OUT is out:
                    r.rowwise_sum[double](X, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    r.rowwise_sum[double](X, OUT)

    def test_colwise_sum(self):
        x_np = np.array([[2., 4., 6.],
                         [1., 3., 5.]])
        expected, expected_t = [3., 7., 11.], [12., 9.]

        self.assertApproxEqual(r.colwise_sum[double](x_np), expected)
        self.assertApproxEqual(r.colwise_sum[double](x_np.T), expected_t)
        out_np = np.empty(3)
        out2 = r.colwise_sum[double](x_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:, :] x = x_np
        self.assertApproxEqual(r.colwise_sum(x), expected)
        out_np[:] = -123.  # reset so that we would catch errors
        cdef double[:] out = out_np
        out2 = r.colwise_sum(x, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_colwise_sum_badargs(self):
        x = np.array([[1.], [2.], [3.]])
        out = np.empty(1)

        for X in (x, np.array([1., 2.]), np.array([[1., 2.]]), None):
            for OUT in (out, np.array([[1.]])):
                if X is x and OUT is out:
                    r.colwise_sum[double](X, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    r.colwise_sum[double](X, OUT)
