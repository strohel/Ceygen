# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np

from support import NoMallocTestCase, malloc_allowed
cimport ceygen.core as c


class TestCore(NoMallocTestCase):

    def test_from_readme(self):
        cdef double[:, :] big = np.array([[1., 2., 2.,  0., 0., 0.],
                                          [3., 4., 0., -2., 0., 0.]])
        self.assertApproxEqual(c.dot_mm(big[:, 0:2], big[:, 2:4], big[:, 4:6]), [[2., -4.], [6., -8.]])
        self.assertApproxEqual(big, [[1., 2., 2.,  0., 2., -4.],
                                     [3., 4., 0., -2., 6., -8.]])
        # TODO: the following line makes Python crash - bug in cython?
        #self.assertApproxEqual(c.dot_mm(big[:, 0:2].T, big[:, 2:4], big[:, 4:6]), [[2., -6.], [4., -8.]])

    def test_eigen_version(self):
        vers = c.eigen_version()
        self.assertTrue(isinstance(vers, tuple))
        self.assertEqual(len(vers), 3)
        self.assertTrue(isinstance(vers[0], int))
        self.assertTrue(isinstance(vers[1], int))
        self.assertTrue(isinstance(vers[2], int))

    def test_dot_vv(self):
        x_np = np.array([1., 2., 3.])
        y_np = np.array([4., 5., 6.])
        self.assertAlmostEqual(c.dot_vv[double](x_np, y_np), 32.)
        cdef double[:] x = x_np
        cdef double[:] y = y_np
        self.assertAlmostEqual(c.dot_vv(x, y), 32.)

    def test_dot_vv_strides(self):
        x = np.array([[1., 2.], [3., 4.]])
        e1 = np.array([1., 0.])
        e2 = np.array([0., 1.])

        self.assertAlmostEqual(c.dot_vv[double](x[0, :], e1), 1.)
        self.assertAlmostEqual(c.dot_vv[double](x[0, :], e2), 2.)
        self.assertAlmostEqual(c.dot_vv[double](x[1, :], e1), 3.)
        self.assertAlmostEqual(c.dot_vv[double](x[1, :], e2), 4.)

        self.assertAlmostEqual(c.dot_vv[double](x[:, 0], e1), 1.)
        self.assertAlmostEqual(c.dot_vv[double](x[:, 0], e2), 3.)
        self.assertAlmostEqual(c.dot_vv[double](x[:, 1], e1), 2.)
        self.assertAlmostEqual(c.dot_vv[double](x[:, 1], e2), 4.)

    def test_dot_vv_baddims(self):
        x = np.array([1., 2., 3.])
        y = np.array([4., 5.])
        z = np.array([[1., 2.], [3., 4.]])
        def dot_vv(x, y):
            # wrap up because c.dot_vv is cython-only (not callable from Python)
            return c.dot_vv[double](x, y)

        self.assertRaises(ValueError, dot_vv, x, y)
        self.assertRaises(ValueError, dot_vv, x, z)

    def test_dot_vv_none(self):
        x = np.array([1., 2., 3.])
        def dot_vv(x, y):
            return c.dot_vv[double](x, y)
        self.assertRaises(ValueError, dot_vv, x, None)
        self.assertRaises(TypeError, dot_vv, x, [1., 2., 3.])
        self.assertRaises(ValueError, dot_vv, None, x)
        self.assertRaises(TypeError, dot_vv, [1., 2., 3.], x)


    def test_dot_mv(self):
        x_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        y_np = np.array([4., 5., 6.])
        self.assertApproxEqual(c.dot_mv[double](x_np, y_np), np.array([32., 28.]))
        self.assertApproxEqual(c.dot_mv[double](x_np, y_np, None), np.array([32., 28.]))
        out_np = np.zeros(2)
        out2_np = c.dot_mv[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, np.array([32., 28.]))  # test that it actually uses out
        self.assertApproxEqual(out2_np, np.array([32., 28.]))

        cdef double[:, :] x = x_np
        cdef double[:] y = y_np
        self.assertApproxEqual(c.dot_mv(x, y), np.array([32., 28.]))
        cdef double[:] out = out_np
        cdef double[:] out2 = c.dot_mv(x, y, out)
        self.assertApproxEqual(out, np.array([32., 28.]))  # test that it actually uses out
        self.assertApproxEqual(out2, np.array([32., 28.]))

    def test_dot_mv_transposed(self):
        x_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        y_np = np.array([4., 5.])
        self.assertApproxEqual(c.dot_mv[double](x_np.T, y_np), np.array([19., 18., 17.]))

    def test_dot_mv_strides(self):
        big = np.array([[[1., 2.], [3., 4.]],
                        [[5., 6.], [7., 8.]]])
        e1 = np.array([1., 0.])
        e2 = np.array([0., 1.])
        self.assertApproxEqual(c.dot_mv[double](big[0, :, :], e1), np.array([1., 3.]))
        self.assertApproxEqual(c.dot_mv[double](big[0, :, :], e2), np.array([2., 4.]))
        self.assertApproxEqual(c.dot_mv[double](big[1, :, :], e1), np.array([5., 7.]))
        self.assertApproxEqual(c.dot_mv[double](big[1, :, :], e2), np.array([6., 8.]))

        self.assertApproxEqual(c.dot_mv[double](big[:, 0, :], e1), np.array([1., 5.]))
        self.assertApproxEqual(c.dot_mv[double](big[:, 0, :], e2), np.array([2., 6.]))
        self.assertApproxEqual(c.dot_mv[double](big[:, 1, :], e1), np.array([3., 7.]))
        self.assertApproxEqual(c.dot_mv[double](big[:, 1, :], e2), np.array([4., 8.]))

        self.assertApproxEqual(c.dot_mv[double](big[:, :, 0], e1), np.array([1., 5.]))
        self.assertApproxEqual(c.dot_mv[double](big[:, :, 0], e2), np.array([3., 7.]))
        self.assertApproxEqual(c.dot_mv[double](big[:, :, 1], e1), np.array([2., 6.]))
        self.assertApproxEqual(c.dot_mv[double](big[:, :, 1], e2), np.array([4., 8.]))

    def test_dot_mv_baddims(self):
        def dot_mv(x, y, out=None):
            return c.dot_mv[double](x, y, out)
        X = np.array([[1., 2., 3.],[2., 3., 4.]])
        y = np.array([1., 2., 3.])
        self.assertRaises(ValueError, dot_mv, np.array([1., 2.]), np.array([1., 2.]))
        self.assertRaises(ValueError, dot_mv, X, np.array([1., 2.]))
        self.assertRaises(ValueError, dot_mv, X, np.array([1.]))
        self.assertRaises(ValueError, dot_mv, X.T, y)

        # good x, y dims, but bad out dims
        self.assertRaises(ValueError, dot_mv, X, y, np.zeros(1))
        self.assertRaises(ValueError, dot_mv, X, y, np.zeros(3))

    def test_dot_mv_none(self):
        x, y, out = np.array([[3.]]), np.array([2.]), np.zeros(1)
        for X in (x, None):
            for Y in (y, None):
                for out in (None, out):
                    if X is x and Y is y:
                        continue  # this case would be valid
                    try:
                        c.dot_mv[double](X, Y, out)
                    except ValueError:
                        pass
                    else:
                        self.fail("ValueError was not raised (X={0}, Y={1}, out={2}".format(X, Y, out))


    def test_dot_vm(self):
        x_np = np.array([4., 5.])
        y_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        expected = np.array([19., 18., 17.])
        self.assertApproxEqual(c.dot_vm[double](x_np, y_np), expected)
        self.assertApproxEqual(c.dot_vm[double](x_np, y_np, None), expected)
        out_np = np.zeros(3)
        out2_np = c.dot_vm[double](x_np, y_np, out_np)
        self.assertApproxEqual(out_np, expected)  # test that it actually uses out
        self.assertApproxEqual(out2_np, expected)

        cdef double[:] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(c.dot_vm(x, y), expected)
        cdef double[:] out = out_np
        cdef double[:] out2 = c.dot_vm(x, y, out)
        self.assertApproxEqual(out, expected)  # test that it actually uses out
        self.assertApproxEqual(out2, expected)

    def test_dot_vm_transposed(self):
        x_np = np.array([4., 5., 6.])
        y_np = np.array([[1., 2., 3.], [3., 2., 1.]])
        self.assertApproxEqual(c.dot_vm[double](x_np, y_np.T), np.array([32., 28.]))

    def test_dot_vm_baddims(self):
        def dot_vm(x, y, out=None):
            return c.dot_vm[double](x, y, out)
        x = np.array([1., 2.])
        y = np.array([[1., 2., 3.],[2., 3., 4.]])
        self.assertRaises(ValueError, dot_vm, np.array([1., 2.]), np.array([1., 2.]))
        self.assertRaises(ValueError, dot_vm, x, np.array([[1., 2.], [2., 3.], [3., 4.]]))
        self.assertRaises(ValueError, dot_vm, np.array([1.]), y)
        self.assertRaises(ValueError, dot_vm, x, y.T)

        # good x, y dims, but bad out dims
        self.assertRaises(ValueError, dot_vm, x, y, np.zeros(1))
        self.assertRaises(ValueError, dot_vm, x, y, np.zeros(2))
        self.assertRaises(ValueError, dot_vm, x, y, np.zeros(4))

    def test_dot_vm_none(self):
        x, y, out = np.array([3.]), np.array([[2.]]), np.zeros(1)
        for X in (x, None):
            for Y in (y, None):
                for out in (None, out):
                    if X is x and Y is y:
                        continue  # this case would be valid
                    try:
                        c.dot_vm[double](X, Y, out)
                    except ValueError:
                        pass
                    else:
                        self.fail("ValueError was not raised (X={0}, Y={1}, out={2}".format(X, Y, out))


    def test_dot_mm(self):
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

        self.assertApproxEqual(c.dot_mm[double](x_np, y_np), expected[0])
        self.assertApproxEqual(c.dot_mm[double](x_np.T, y_np), expected[1])
        self.assertApproxEqual(c.dot_mm[double](x_np, y_np.T), expected[2])
        self.assertApproxEqual(c.dot_mm[double](x_np.T, y_np.T), expected[3])

        cdef double[:, :] x = x_np
        cdef double[:, :] y = y_np
        self.assertApproxEqual(c.dot_mm(x, y), expected[0])
        self.assertApproxEqual(c.dot_mm(x.T, y), expected[1])
        self.assertApproxEqual(c.dot_mm(x, y.T), expected[2])
        self.assertApproxEqual(c.dot_mm(x.T, y.T), expected[3])

        # test that it actually uses out
        out_np = np.empty((2, 2))
        cdef double[:, :] out = out_np
        out2 = c.dot_mm(x, y, out)
        self.assertApproxEqual(out2, expected[0])
        self.assertApproxEqual(out, expected[0])
        self.assertApproxEqual(out_np, expected[0])

        a_np = np.array([[1., 2., 3.], [4., 5., 6.]])
        b_np = np.array([[1.], [2.], [3.]])
        self.assertApproxEqual(c.dot_mm[double](a_np, b_np), np.array([[14.], [32.]]))
        self.assertApproxEqual(c.dot_mm[double](b_np.T, a_np.T), np.array([[14., 32.]]))
        cdef double[:, :] a = a_np
        cdef double[:, :] b = b_np
        self.assertApproxEqual(c.dot_mm(a, b), np.array([[14.], [32.]]))
        self.assertApproxEqual(c.dot_mm(b.T, a.T), np.array([[14., 32.]]))

    def test_dot_mm_strides(self):
        big = np.array([[[1., 2.], [3., 4.]],
                        [[5., 6.], [7., 8.]]])
        eye = np.eye(2)

        # following are still C-contiguous:
        self.assertApproxEqual(c.dot_mm[double](big[0, :, :], eye), big[0, :, :])
        self.assertApproxEqual(c.dot_mm[double](big[1, :, :], eye), big[1, :, :])
        self.assertApproxEqual(c.dot_mm[double](big[:, 0, :], eye), big[:, 0, :])
        self.assertApproxEqual(c.dot_mm[double](big[:, 1, :], eye), big[:, 1, :])

        # following are Fortran-contiguous:
        self.assertApproxEqual(c.dot_mm[double](big[0, :, :].T, eye), big[0, :, :].T)
        self.assertApproxEqual(c.dot_mm[double](big[1, :, :].T, eye), big[1, :, :].T)
        self.assertApproxEqual(c.dot_mm[double](big[:, 0, :].T, eye), big[:, 0, :].T)
        self.assertApproxEqual(c.dot_mm[double](big[:, 1, :].T, eye), big[:, 1, :].T)

        # actually test that our infractructure is capable of detecting memory allocations
        for myslice in (big[:, :, 0], big[:, :, 1], big[:, :, 1].T, big[:, :, 1].T):
            with self.assertRaises(ValueError):
                c.dot_mm[double](myslice, eye)

        # non-contiguous slices in dot_mm cause memory allocations in Eigen, expect it:
        with malloc_allowed():
            self.assertApproxEqual(c.dot_mm[double](big[:, :, 0], eye), big[:, :, 0])
            self.assertApproxEqual(c.dot_mm[double](big[:, :, 1], eye), big[:, :, 1])
            self.assertApproxEqual(c.dot_mm[double](big[:, :, 0].T, eye), big[:, :, 0].T)
            self.assertApproxEqual(c.dot_mm[double](big[:, :, 1].T, eye), big[:, :, 1].T)

        # assert that we've reenabled assertions on memory allocations
        with self.assertRaises(ValueError):
            c.dot_mm[double](big[:, :, 0], eye)

    def test_dot_mm_baddims(self):
        x = np.array([[1., 2.],
                      [3., 4.]])
        y = np.array([[5., 6.],
                      [7., 8.]])
        out = np.empty((2, 2))
        for X in (x, np.array([1., 2.]), np.array([[1.], [2.]]), np.array([[[1.]]])):
            for Y in (y, np.array([1., 2.]), np.array([[1.], [2.], [3.]]), np.array([[[1.]]])):
                for OUT in (out, None, np.empty((2,)), np.empty((2, 3)), np.empty((3, 2)), np.empty((2, 2, 1))):
                    if X is x and Y is y and (OUT is out or OUT is None):
                        continue  # these would be valid
                    try:
                        c.dot_mm[double](X, Y, OUT)
                    except ValueError:
                        pass
                    else:
                        self.fail("ValueError was not raised (X={0}, Y={1}, OUT={2}".format(X, Y, OUT))

    def test_dot_mm_none(self):
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
                        c.dot_mm[double](X, Y, OUT)
                    except ValueError:
                        pass
                    else:
                        self.fail("ValueError was not raised (X={0}, Y={1}, OUT={2}".format(X, Y, OUT))
