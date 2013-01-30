# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np

from support import CeygenTestCase
cimport ceygen.elemwise as e


class TestElemwise(CeygenTestCase):

    def test_add_vv(self):
        x_np = np.array([1., 3., 5.])
        y_np = np.array([3., 2., 1.])
        expected_xy, expected_yx = np.array([4., 5., 6.]), np.array([4., 5., 6.])

        self.assertApproxEqual(e.add_vv(x_np, y_np), expected_xy)
        self.assertApproxEqual(e.add_vv(y_np, x_np), expected_yx)
        out_np = np.empty(3)
        out2 = e.add_vv(x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(e.add_vv(x, y), expected_xy)
        self.assertApproxEqual(e.add_vv(y, x), expected_yx)
        out = out_np
        out2 = e.add_vv(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_add_vv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for Y in (y, np.array([1., 2., 3., 4.])):
                for OUT in (out, np.empty(2), np.empty(4)):
                    if X is x and Y is y and OUT is out:
                        e.add_vv(X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.add_vv(X, Y, OUT)

    def test_add_vv_none(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.add_vv(X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.add_vv(X, Y)


    def test_subtract_vv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([3., 2., 1.])
        expected_xy, expected_yx = np.array([-2., 0., 2.]), np.array([2., 0., -2.])

        self.assertApproxEqual(e.subtract_vv(x_np, y_np), expected_xy)
        self.assertApproxEqual(e.subtract_vv(y_np, x_np), expected_yx)
        out_np = np.empty(3)
        out2 = e.subtract_vv(x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(e.subtract_vv(x, y), expected_xy)
        self.assertApproxEqual(e.subtract_vv(y, x), expected_yx)
        out = out_np
        out2 = e.subtract_vv(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_subtract_vv_bsubtractims(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for Y in (y, np.array([1., 2., 3., 4.])):
                for OUT in (out, np.empty(2), np.empty(4)):
                    if X is x and Y is y and OUT is out:
                        e.subtract_vv(X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.subtract_vv(X, Y, OUT)

    def test_subtract_vv_none(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.subtract_vv(X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.subtract_vv(X, Y)


    def test_add_mm(self):
        x_np = np.array([[1., 3., 5.]])
        y_np = np.array([[3., 2., 1.]])
        expected_xy, expected_yx = np.array([[4., 5., 6.]]), np.array([[4., 5., 6.]])

        self.assertApproxEqual(e.add_mm(x_np, y_np), expected_xy)
        self.assertApproxEqual(e.add_mm(y_np, x_np), expected_yx)
        self.assertApproxEqual(e.add_mm(x_np.T, y_np.T), expected_xy.T)
        self.assertApproxEqual(e.add_mm(y_np.T, x_np.T), expected_yx.T)
        out_np = np.empty((1, 3))
        out2 = e.add_mm(x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(e.add_mm(x, y), expected_xy)
        self.assertApproxEqual(e.add_mm(y, x), expected_yx)
        self.assertApproxEqual(e.add_mm(x.T, y.T), expected_xy.T)
        self.assertApproxEqual(e.add_mm(y.T, x.T), expected_yx.T)
        out = out_np
        out2 = e.add_mm(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_add_mm_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([[1.], [2.], [3.]])):
            for Y in (y, np.array([[1., 2., 3., 4.]]), np.array([[1.], [2.], [3.], [4.]])):
                for OUT in (out, np.empty((1, 2)), np.empty((3, 1)), np.empty((1, 4))):
                    if X is x and Y is y and OUT is out:
                        e.add_mm(X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.add_mm(X, Y, OUT)

    def test_add_mm_none(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.add_mm(X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.add_mm(X, Y)


    def test_subtract_mm(self):
        x_np = np.array([[1., 2., 3.]])
        y_np = np.array([[3., 2., 1.]])
        expected_xy, expected_yx = np.array([[-2., 0., 2.]]), np.array([[2., 0., -2.]])

        self.assertApproxEqual(e.subtract_mm(x_np, y_np), expected_xy)
        self.assertApproxEqual(e.subtract_mm(y_np, x_np), expected_yx)
        self.assertApproxEqual(e.subtract_mm(x_np.T, y_np.T), expected_xy.T)
        self.assertApproxEqual(e.subtract_mm(y_np.T, x_np.T), expected_yx.T)
        out_np = np.empty((1, 3))
        out2 = e.subtract_mm(x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(e.subtract_mm(x, y), expected_xy)
        self.assertApproxEqual(e.subtract_mm(y, x), expected_yx)
        self.assertApproxEqual(e.subtract_mm(x.T, y.T), expected_xy.T)
        self.assertApproxEqual(e.subtract_mm(y.T, x.T), expected_yx.T)
        out = out_np
        out2 = e.subtract_mm(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_subtract_mm_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([[1.], [2.], [3.]])):
            for Y in (y, np.array([[1., 2., 3., 4.]]), np.array([[1.], [2.], [3.], [4.]])):
                for OUT in (out, np.empty((1, 2)), np.empty((3, 1)), np.empty((1, 4))):
                    if X is x and Y is y and OUT is out:
                        e.subtract_mm(X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.subtract_mm(X, Y, OUT)

    def test_subtract_mm_none(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.subtract_mm(X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.subtract_mm(X, Y)
