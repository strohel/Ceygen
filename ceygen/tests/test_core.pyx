# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

"""Tests for core"""

import numpy as np

from support import CeygenTestCase
cimport ceygen.core as c

class TestCore(CeygenTestCase):

    def test_dotvv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([4., 5., 6.])
        self.assertAlmostEqual(c.dotvv(x_np, y_np), 32.)
        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertAlmostEqual(c.dotvv(x, y), 32.)

    def test_dotvv_strides(self):
        x = np.array([[1., 2.], [3., 4.]])
        e1 = np.array([1., 0.])
        e2 = np.array([0., 1.])

        self.assertAlmostEqual(c.dotvv(x[0, :], e1), 1.)
        self.assertAlmostEqual(c.dotvv(x[0, :], e2), 2.)
        self.assertAlmostEqual(c.dotvv(x[1, :], e1), 3.)
        self.assertAlmostEqual(c.dotvv(x[1, :], e2), 4.)

        self.assertAlmostEqual(c.dotvv(x[:, 0], e1), 1.)
        self.assertAlmostEqual(c.dotvv(x[:, 0], e2), 3.)
        self.assertAlmostEqual(c.dotvv(x[:, 1], e1), 2.)
        self.assertAlmostEqual(c.dotvv(x[:, 1], e2), 4.)

    def test_dotvv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([4., 5.])
        z = np.array([[1., 2.], [3., 4.]])
        def dotvv(x, y):
            # wrap up because c.dotvv is cython-only (not callable from Python)
            return c.dotvv(x, y)

        self.assertRaises(StandardError, dotvv, x, y)
        self.assertRaises(StandardError, dotvv, x, z)

    def test_dotvv_none(self):
        x = np.array([1., 2., 3.])
        def dotvv(x, y):
            return c.dotvv(x, y)
        self.assertRaises(StandardError, dotvv, x, None)
        self.assertRaises(StandardError, dotvv, x, "Hello")
        self.assertRaises(StandardError, dotvv, None, x)
        self.assertRaises(StandardError, dotvv, "Hello", x)


    def test_dotmv(self):
        x_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        y_np = np.array([4., 5., 6.])
        self.assertApproxEqual(c.dotmv(x_np, y_np), np.array([32., 28.]))
        self.assertApproxEqual(c.dotmv(x_np, y_np, None), np.array([32., 28.]))
        out_np = np.zeros(2)
        out2_np = c.dotmv(x_np, y_np, out_np)
        self.assertApproxEqual(out_np, np.array([32., 28.]))  # test that it actually uses out
        self.assertApproxEqual(out2_np, np.array([32., 28.]))

        cdef double[:, :] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(c.dotmv(x, y), np.array([32., 28.]))
        cdef double[:] out = out_np
        cdef double[:] out2 = c.dotmv(x_np, y_np, out)
        self.assertApproxEqual(out, np.array([32., 28.]))  # test that it actually uses out
        self.assertApproxEqual(out2, np.array([32., 28.]))

    def test_dotmv_transposed(self):
        x_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        y_np = np.array([4., 5.])
        self.assertApproxEqual(c.dotmv(x_np.T, y_np), np.array([19., 18., 17.]))

    def test_dotmv_strides(self):
        big = np.array([[[1., 2.], [3., 4.]],
                        [[5., 6.], [7., 8.]]])
        e1 = np.array([1., 0.])
        e2 = np.array([0., 1.])
        self.assertApproxEqual(c.dotmv(big[0, :, :], e1), np.array([1., 3.]))
        self.assertApproxEqual(c.dotmv(big[0, :, :], e2), np.array([2., 4.]))
        self.assertApproxEqual(c.dotmv(big[1, :, :], e1), np.array([5., 7.]))
        self.assertApproxEqual(c.dotmv(big[1, :, :], e2), np.array([6., 8.]))

        self.assertApproxEqual(c.dotmv(big[:, 0, :], e1), np.array([1., 5.]))
        self.assertApproxEqual(c.dotmv(big[:, 0, :], e2), np.array([2., 6.]))
        self.assertApproxEqual(c.dotmv(big[:, 1, :], e1), np.array([3., 7.]))
        self.assertApproxEqual(c.dotmv(big[:, 1, :], e2), np.array([4., 8.]))

        self.assertApproxEqual(c.dotmv(big[:, :, 0], e1), np.array([1., 5.]))
        self.assertApproxEqual(c.dotmv(big[:, :, 0], e2), np.array([3., 7.]))
        self.assertApproxEqual(c.dotmv(big[:, :, 1], e1), np.array([2., 6.]))
        self.assertApproxEqual(c.dotmv(big[:, :, 1], e2), np.array([4., 8.]))

    def test_dotmv_baddims(self):
        def dotmv(x, y, out=None):
            return c.dotmv(x, y, out)
        X = np.array([[1., 2., 3.],[2., 3., 4.]])
        y = np.array([1., 2., 3.])
        self.assertRaises(StandardError, dotmv, np.array([1., 2.]), np.array([1., 2.]))
        self.assertRaises(StandardError, dotmv, X, np.array([1., 2.]))
        self.assertRaises(StandardError, dotmv, X, np.array([1.]))
        self.assertRaises(StandardError, dotmv, X.T, y)

        # good x, y dims, but bad out dims
        self.assertRaises(StandardError, dotmv, X, y, np.zeros(1))
        self.assertRaises(StandardError, dotmv, X, y, np.zeros(3))

    def test_dotmv_none(self):
        x, y, out = np.array([[3.]]), np.array([2.]), np.zeros(1)
        for X in (x, None):
            for Y in (y, None):
                for out in (None, out):
                    if X is x and Y is y:
                        continue  # this case would be valid
                    try:
                        c.dotmv(X, Y, out)
                    except StandardError as e:
                        #print e
                        pass
                    else:
                        self.fail("StandardError was not raised (X={0}, Y={1}, out={2}".format(X, Y, out))


    def test_dotvm(self):
        x_np = np.array([4., 5.])
        y_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        expected = np.array([19., 18., 17.])
        self.assertApproxEqual(c.dotvm(x_np, y_np), expected)
        self.assertApproxEqual(c.dotvm(x_np, y_np, None), expected)
        out_np = np.zeros(3)
        out2_np = c.dotvm(x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2_np, expected)

        cdef double[:] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(c.dotvm(x, y), expected)
        cdef double[:] out = out_np
        cdef double[:] out2 = c.dotvm(x_np, y_np, out)
        self.assertApproxEqual(out, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

    def test_dotvm_transposed(self):
        x_np = np.array([4., 5., 6.])
        y_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        self.assertApproxEqual(c.dotvm(x_np, y_np.T), np.array([32., 28.]))

    def test_dotvm_baddims(self):
        def dotvm(x, y, out=None):
            return c.dotvm(x, y, out)
        x = np.array([1., 2.])
        y = np.array([[1., 2., 3.],[2., 3., 4.]])
        self.assertRaises(StandardError, dotvm, np.array([1., 2.]), np.array([1., 2.]))
        self.assertRaises(StandardError, dotvm, x, np.array([[1., 2.], [2., 3.], [3., 4.]]))
        self.assertRaises(StandardError, dotvm, np.array([1.]), y)
        self.assertRaises(StandardError, dotvm, x, y.T)

        # good x, y dims, but bad out dims
        self.assertRaises(StandardError, dotvm, x, y, np.zeros(1))
        self.assertRaises(StandardError, dotvm, x, y, np.zeros(2))
        self.assertRaises(StandardError, dotvm, x, y, np.zeros(4))

    def test_dotvm_none(self):
        x, y, out = np.array([3.]), np.array([[2.]]), np.zeros(1)
        for X in (x, None):
            for Y in (y, None):
                for out in (None, out):
                    if X is x and Y is y:
                        continue  # this case would be valid
                    try:
                        c.dotvm(X, Y, out)
                    except StandardError:
                        pass
                    else:
                        self.fail("StandardError was not raised (X={0}, Y={1}, out={2}".format(X, Y, out))

    def test_dotmm(self):
        x_np = np.array([[1., 2.],
                         [3., 4.]])
        y_np = np.array([[5., 6.],
                         [7., 8.]])
        expected = [
            np.array([[19., 22.], [43., 50.]]),
            np.array([[26., 30.], [38., 44.]]),
            np.array([[17., 23.], [39., 53.]]),
            np.array([[23., 31.], [34., 46.]])
        ]

        self.assertApproxEqual(c.dotmm(x_np, y_np), expected[0])
        self.assertApproxEqual(c.dotmm(x_np.T, y_np), expected[1])
        self.assertApproxEqual(c.dotmm(x_np, y_np.T), expected[2])
        self.assertApproxEqual(c.dotmm(x_np.T, y_np.T), expected[3])

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(c.dotmm(x, y), expected[0])
        self.assertApproxEqual(c.dotmm(x.T, y), expected[1])
        self.assertApproxEqual(c.dotmm(x, y.T), expected[2])
        self.assertApproxEqual(c.dotmm(x.T, y.T), expected[3])

        # test that it actually uses out
        out_np = np.empty((2, 2))
        cdef double[:, :] out = out_np
        out2 = c.dotmm(x, y, out)
        self.assertApproxEqual(out2, expected[0])
        self.assertApproxEqual(out, expected[0])
        self.assertApproxEqual(out_np, expected[0])

    def test_dotmm_baddims(self):
        x = np.array([[1., 2.],
                      [3., 4.]])
        y = np.array([[5., 6.],
                      [7., 8.]])
        out = np.empty((2, 2))
        for X in (x, np.array([1., 2.]), np.array([[1.], [2.]]), np.array([[[1.]]])):
            for Y in (y, np.array([1., 2.]), np.array([[1.], [2.]]), np.array([[[1.]]])):
                for OUT in (out, None, np.empty((2,)), np.empty((2, 3)), np.empty((3, 2)), np.empty((2, 2, 1))):
                    if X is x and Y is y and (OUT is out or OUT is None):
                        continue  # these would be valid
                    try:
                        c.dotmm(X, Y, OUT)
                    except StandardError:
                        pass
                    else:
                        self.fail("StandardError was not raised (X={0}, Y={1}, OUT={2}".format(X, Y, OUT))

    def test_dotmm_none(self):
        x = np.array([[1., 2.],
                      [3., 4.]])
        y = np.array([[5., 6.],
                      [7., 8.]])
        out = np.empty((2, 2))
        for X in (x, None):
            for Y in (y, None):
                for OUT in (out, None):
                    if X is x and Y in y:
                        continue  # this would be valid
                    try:
                        c.dotmm(X, Y, OUT)
                    except StandardError:
                        pass
                    else:
                        self.fail("StandardError was not raised (X={0}, Y={1}, OUT={2}".format(X, Y, OUT))
