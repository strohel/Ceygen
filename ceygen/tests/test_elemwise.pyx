# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np

from support import NoMallocTestCase
cimport ceygen.elemwise as e


class TestElemwise(NoMallocTestCase):

    def test_add_vs(self):
        x_np = np.array([1., 3., 5.])
        y_np = -4.
        expected = np.array([-3., -1., 1.])

        self.assertApproxEqual(e.add_vs[double](x_np, y_np), expected)
        out_np = np.empty(3)
        out2 = e.add_vs[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:] x = x_np
        cdef double y = y_np
        self.assertApproxEqual(e.add_vs(x, y), expected)
        out_np[:] = -123.  # reset so that we would catch errors
        cdef double[:] out = out_np
        out2 = e.add_vs(x, y, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_add_vs_baddims(self):
        x = np.array([1., 2., 3.])
        y = 3.
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for OUT in (out, np.empty(1), np.empty(4)):
                if X is x and OUT is out:
                    e.add_vs[double](X, y, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.add_vs[double](X, y, OUT)

    def test_add_vs_none(self):
        with self.assertRaises(ValueError):
            e.add_vs[double](None, 3.)


    def test_multiply_vs(self):
        x_np = np.array([1., 3., -5.])
        y_np = -4.
        expected = np.array([-4., -12., 20.])

        self.assertApproxEqual(e.multiply_vs[double](x_np, y_np), expected)
        out_np = np.empty(3)
        out2 = e.multiply_vs[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:] x = x_np
        cdef double y = y_np
        self.assertApproxEqual(e.multiply_vs(x, y), expected)
        out_np[:] = -123.  # reset so that we would catch errors
        cdef double[:] out = out_np
        out2 = e.multiply_vs(x, y, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_multiply_vs_baddims(self):
        x = np.array([1., 2., 3.])
        y = 3.
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for OUT in (out, np.empty(1), np.empty(4)):
                if X is x and OUT is out:
                    e.multiply_vs[double](X, y, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.multiply_vs[double](X, y, OUT)

    def test_multiply_vs_none(self):
        with self.assertRaises(ValueError):
            e.multiply_vs[double](None, 3.)


    def test_power_vs(self):
        x_np = np.array([1., 3., -5.])
        y_np = 2.
        expected = np.array([1., 9., 25.])

        self.assertApproxEqual(e.power_vs[double](x_np, y_np), expected)
        out_np = np.empty(3)
        out2 = e.power_vs[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:] x = x_np
        cdef double y = y_np
        self.assertApproxEqual(e.power_vs(x, y), expected)
        out_np[:] = -123.  # reset so that we would catch errors
        cdef double[:] out = out_np
        out2 = e.power_vs(x, y, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_power_vs_baddims(self):
        x = np.array([1., 2., 3.])
        y = 3.
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for OUT in (out, np.empty(1), np.empty(4)):
                if X is x and OUT is out:
                    e.power_vs[double](X, y, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.power_vs[double](X, y, OUT)

    def test_power_vs_none(self):
        with self.assertRaises(ValueError):
            e.power_vs[double](None, 3.)


    def test_add_vv(self):
        x_np = np.array([1., 3., 5.])
        y_np = np.array([3., 2., 1.])
        expected_xy, expected_yx = np.array([4., 5., 6.]), np.array([4., 5., 6.])

        self.assertApproxEqual(e.add_vv[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.add_vv[double](y_np, x_np), expected_yx)
        out_np = np.empty(3)
        out2 = e.add_vv[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(e.add_vv(x, y), expected_xy)
        self.assertApproxEqual(e.add_vv(y, x), expected_yx)
        cdef double[:] out = out_np
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
                        e.add_vv[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.add_vv[double](X, Y, OUT)

    def test_add_vv_none(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.add_vv[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.add_vv[double](X, Y)


    def test_subtract_vv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([3., 2., 1.])
        expected_xy, expected_yx = np.array([-2., 0., 2.]), np.array([2., 0., -2.])

        self.assertApproxEqual(e.subtract_vv[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.subtract_vv[double](y_np, x_np), expected_yx)
        out_np = np.empty(3)
        out2 = e.subtract_vv[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(e.subtract_vv(x, y), expected_xy)
        self.assertApproxEqual(e.subtract_vv(y, x), expected_yx)
        cdef double[:] out = out_np
        out2 = e.subtract_vv(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_subtract_vv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for Y in (y, np.array([1., 2., 3., 4.])):
                for OUT in (out, np.empty(2), np.empty(4)):
                    if X is x and Y is y and OUT is out:
                        e.subtract_vv[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.subtract_vv[double](X, Y, OUT)

    def test_subtract_vv_none(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.subtract_vv[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.subtract_vv[double](X, Y)


    def test_multiply_vv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([3., 2., 1.])
        expected_xy, expected_yx = np.array([3., 4., 3.]), np.array([3., 4., 3.])

        self.assertApproxEqual(e.multiply_vv[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.multiply_vv[double](y_np, x_np), expected_yx)
        out_np = np.empty(3)
        out2 = e.multiply_vv[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(e.multiply_vv(x, y), expected_xy)
        self.assertApproxEqual(e.multiply_vv(y, x), expected_yx)
        cdef double[:] out = out_np
        out2 = e.multiply_vv(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_multiply_vv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for Y in (y, np.array([1., 2., 3., 4.])):
                for OUT in (out, np.empty(2), np.empty(4)):
                    if X is x and Y is y and OUT is out:
                        e.multiply_vv[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.multiply_vv[double](X, Y, OUT)

    def test_multiply_vv_none(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.multiply_vv[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.multiply_vv[double](X, Y)


    def test_divide_vv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([3., 2., 1.])
        expected_xy, expected_yx = np.array([1./3., 1., 3.]), np.array([3., 1., 1./3.])

        self.assertApproxEqual(e.divide_vv[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.divide_vv[double](y_np, x_np), expected_yx)
        out_np = np.empty(3)
        out2 = e.divide_vv[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(e.divide_vv(x, y), expected_xy)
        self.assertApproxEqual(e.divide_vv(y, x), expected_yx)
        cdef double[:] out = out_np
        out2 = e.divide_vv(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_divide_vv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])
        out = np.empty(3)

        for X in (x, np.array([1., 2.])):
            for Y in (y, np.array([1., 2., 3., 4.])):
                for OUT in (out, np.empty(2), np.empty(4)):
                    if X is x and Y is y and OUT is out:
                        e.divide_vv[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.divide_vv[double](X, Y, OUT)

    def test_divide_vv_none(self):
        x = np.array([1., 2., 3.])
        y = np.array([3., 2., 1.])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.divide_vv[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.divide_vv[double](X, Y)


    def test_add_ms(self):
        x_np = np.array([[1., 3., 5.]])
        y_np = -4.
        expected = np.array([[-3., -1., 1.]])

        self.assertApproxEqual(e.add_ms[double](x_np, y_np), expected)
        self.assertApproxEqual(e.add_ms[double](x_np.T, y_np), expected.T)
        out_np = np.empty((1, 3))
        out2 = e.add_ms[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:, :] x = x_np
        cdef double y = y_np
        self.assertApproxEqual(e.add_ms(x, y), expected)
        out_np[:, :] = -123.  # reset so that we would catch errors
        cdef double[:, :] out = out_np
        out2 = e.add_ms(x, y, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_add_ms_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = 3.
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([1., 2.])):
            for OUT in (out, np.empty((1, 1)), np.empty((1, 4)), np.empty(3)):
                if X is x and OUT is out:
                    e.add_ms[double](X, y, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.add_ms[double](X, y, OUT)

    def test_add_ms_none(self):
        with self.assertRaises(ValueError):
            e.add_ms[double](None, 3.)


    def test_multiply_ms(self):
        x_np = np.array([[1., 3., -5.]])
        y_np = -4.
        expected = np.array([[-4., -12., 20.]])

        self.assertApproxEqual(e.multiply_ms[double](x_np, y_np), expected)
        self.assertApproxEqual(e.multiply_ms[double](x_np.T, y_np), expected.T)
        out_np = np.empty((1, 3))
        out2 = e.multiply_ms[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:, :] x = x_np
        cdef double y = y_np
        self.assertApproxEqual(e.multiply_ms(x, y), expected)
        out_np[:, :] = -123.  # reset so that we would catch errors
        cdef double[:, :] out = out_np
        out2 = e.multiply_ms(x, y, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_multiply_ms_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = 3.
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([1., 2.])):
            for OUT in (out, np.empty((1, 1)), np.empty((1, 4)), np.empty(3)):
                if X is x and OUT is out:
                    e.multiply_ms[double](X, y, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.multiply_ms[double](X, y, OUT)

    def test_multiply_ms_none(self):
        with self.assertRaises(ValueError):
            e.multiply_ms[double](None, 3.)


    def test_power_ms(self):
        x_np = np.array([[1., 3., -5.]])
        y_np = 2.
        expected = np.array([[1., 9., 25.]])

        self.assertApproxEqual(e.power_ms[double](x_np, y_np), expected)
        self.assertApproxEqual(e.power_ms[double](x_np.T, y_np), expected.T)
        out_np = np.empty((1, 3))
        out2 = e.power_ms[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

        cdef double[:, :] x = x_np
        cdef double y = y_np
        self.assertApproxEqual(e.power_ms(x, y), expected)
        out_np[:, :] = -123.  # reset so that we would catch errors
        cdef double[:, :] out = out_np
        out2 = e.power_ms(x, y, out)
        self.assertApproxEqual(out, expected)
        self.assertApproxEqual(out2, expected)

    def test_power_ms_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = 3.
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([1., 2.])):
            for OUT in (out, np.empty((1, 1)), np.empty((1, 4)), np.empty(3)):
                if X is x and OUT is out:
                    e.power_ms[double](X, y, OUT)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.power_ms[double](X, y, OUT)

    def test_power_ms_none(self):
        with self.assertRaises(ValueError):
            e.power_ms[double](None, 3.)


    def test_add_mm(self):
        x_np = np.array([[1., 3., 5.]])
        y_np = np.array([[3., 2., 1.]])
        expected_xy, expected_yx = np.array([[4., 5., 6.]]), np.array([[4., 5., 6.]])

        self.assertApproxEqual(e.add_mm[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.add_mm[double](y_np, x_np), expected_yx)
        self.assertApproxEqual(e.add_mm[double](x_np.T, y_np.T), expected_xy.T)
        self.assertApproxEqual(e.add_mm[double](y_np.T, x_np.T), expected_yx.T)
        out_np = np.empty((1, 3))
        out2 = e.add_mm[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(e.add_mm(x, y), expected_xy)
        self.assertApproxEqual(e.add_mm(y, x), expected_yx)
        self.assertApproxEqual(e.add_mm(x.T, y.T), expected_xy.T)
        self.assertApproxEqual(e.add_mm(y.T, x.T), expected_yx.T)
        cdef double[:, :] out = out_np
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
                        e.add_mm[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.add_mm[double](X, Y, OUT)

    def test_add_mm_none(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.add_mm[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.add_mm[double](X, Y)


    def test_subtract_mm(self):
        x_np = np.array([[1., 2., 3.]])
        y_np = np.array([[3., 2., 1.]])
        expected_xy, expected_yx = np.array([[-2., 0., 2.]]), np.array([[2., 0., -2.]])

        self.assertApproxEqual(e.subtract_mm[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.subtract_mm[double](y_np, x_np), expected_yx)
        self.assertApproxEqual(e.subtract_mm[double](x_np.T, y_np.T), expected_xy.T)
        self.assertApproxEqual(e.subtract_mm[double](y_np.T, x_np.T), expected_yx.T)
        out_np = np.empty((1, 3))
        out2 = e.subtract_mm[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(e.subtract_mm(x, y), expected_xy)
        self.assertApproxEqual(e.subtract_mm(y, x), expected_yx)
        self.assertApproxEqual(e.subtract_mm(x.T, y.T), expected_xy.T)
        self.assertApproxEqual(e.subtract_mm(y.T, x.T), expected_yx.T)
        cdef double[:, :] out = out_np
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
                        e.subtract_mm[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.subtract_mm[double](X, Y, OUT)

    def test_subtract_mm_none(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.subtract_mm[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.subtract_mm[double](X, Y)


    def test_multiply_mm(self):
        x_np = np.array([[1., 2., 3.]])
        y_np = np.array([[3., 2., 1.]])
        expected_xy, expected_yx = np.array([[3., 4., 3.]]), np.array([[3., 4., 3.]])

        self.assertApproxEqual(e.multiply_mm[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.multiply_mm[double](y_np, x_np), expected_yx)
        self.assertApproxEqual(e.multiply_mm[double](x_np.T, y_np.T), expected_xy.T)
        self.assertApproxEqual(e.multiply_mm[double](y_np.T, x_np.T), expected_yx.T)
        out_np = np.empty((1, 3))
        out2 = e.multiply_mm[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(e.multiply_mm(x, y), expected_xy)
        self.assertApproxEqual(e.multiply_mm(y, x), expected_yx)
        self.assertApproxEqual(e.multiply_mm(x.T, y.T), expected_xy.T)
        self.assertApproxEqual(e.multiply_mm(y.T, x.T), expected_yx.T)
        cdef double[:, :] out = out_np
        out2 = e.multiply_mm(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_multiply_mm_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([[1.], [2.], [3.]])):
            for Y in (y, np.array([[1., 2., 3., 4.]]), np.array([[1.], [2.], [3.], [4.]])):
                for OUT in (out, np.empty((1, 2)), np.empty((3, 1)), np.empty((1, 4))):
                    if X is x and Y is y and OUT is out:
                        e.multiply_mm[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.multiply_mm[double](X, Y, OUT)

    def test_multiply_mm_none(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.multiply_mm[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.multiply_mm[double](X, Y)


    def test_divide_mm(self):
        x_np = np.array([[1., 2., 3.]])
        y_np = np.array([[3., 2., 1.]])
        expected_xy, expected_yx = np.array([[1./3., 1., 3.]]), np.array([[3., 1., 1./3.]])

        self.assertApproxEqual(e.divide_mm[double](x_np, y_np), expected_xy)
        self.assertApproxEqual(e.divide_mm[double](y_np, x_np), expected_yx)
        self.assertApproxEqual(e.divide_mm[double](x_np.T, y_np.T), expected_xy.T)
        self.assertApproxEqual(e.divide_mm[double](y_np.T, x_np.T), expected_yx.T)
        out_np = np.empty((1, 3))
        out2 = e.divide_mm[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected_xy)  # test that it actually uses out
        self.assertApproxEqual(out2, expected_xy)

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(e.divide_mm(x, y), expected_xy)
        self.assertApproxEqual(e.divide_mm(y, x), expected_yx)
        self.assertApproxEqual(e.divide_mm(x.T, y.T), expected_xy.T)
        self.assertApproxEqual(e.divide_mm(y.T, x.T), expected_yx.T)
        cdef double[:, :] out = out_np
        out2 = e.divide_mm(y, x, out)
        self.assertApproxEqual(out, expected_yx)
        self.assertApproxEqual(out2, expected_yx)

    def test_divide_mm_baddims(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])
        out = np.empty((1, 3))

        for X in (x, np.array([[1., 2.]]), np.array([[1.], [2.], [3.]])):
            for Y in (y, np.array([[1., 2., 3., 4.]]), np.array([[1.], [2.], [3.], [4.]])):
                for OUT in (out, np.empty((1, 2)), np.empty((3, 1)), np.empty((1, 4))):
                    if X is x and Y is y and OUT is out:
                        e.divide_mm[double](X, Y, OUT)  # this should be valid
                        continue
                    with self.assertRaises(ValueError):
                        e.divide_mm[double](X, Y, OUT)

    def test_divide_mm_none(self):
        x = np.array([[1., 2., 3.]])
        y = np.array([[3., 2., 1.]])

        for X in (x, None):
            for Y in (y, None):
                if X is x and Y is y:
                    e.divide_mm[double](X, Y)  # this should be valid
                    continue
                with self.assertRaises(ValueError):
                    e.divide_mm[double](X, Y)
