# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cimport cython

from support import CeygenTestCase
cimport ceygen.dtype as dtype


class TestDtype(CeygenTestCase):

    def test_vector(self):
        cdef char[:] c
        cdef short[:] h
        cdef int[:] i
        cdef long[:] l
        cdef float[:] f
        cdef double[:] d

        cdef int length
        for length in (1, 3, 16, 134):  # TODO: 0 should be valid, but Cython doesn't like it
            c = dtype.vector(length, <char *> 0)
            h = dtype.vector(length, <short *> 0)
            i = dtype.vector(length, <int *> 0)
            l = dtype.vector(length, <long *> 0)
            f = dtype.vector(length, <float *> 0)
            d = dtype.vector(length, <double *> 0)
            self.assertEqual(c.shape[0], length)
            self.assertEqual(h.shape[0], length)
            self.assertEqual(i.shape[0], length)
            self.assertEqual(l.shape[0], length)
            self.assertEqual(f.shape[0], length)
            self.assertEqual(d.shape[0], length)

    def test_matrix(self):
        cdef char[:, :] c
        cdef short[:, :] h
        cdef int[:, :] i
        cdef long[:, :] l
        cdef float[:, :] f
        cdef double[:, :] d

        cdef int rows, cols
        for rows in (1, 3, 16, 134):
            for cols in (1, 2, 16, 209):
                c = dtype.matrix(rows, cols, <char *> 0)
                h = dtype.matrix(rows, cols, <short *> 0)
                i = dtype.matrix(rows, cols, <int *> 0)
                l = dtype.matrix(rows, cols, <long *> 0)
                f = dtype.matrix(rows, cols, <float *> 0)
                d = dtype.matrix(rows, cols, <double *> 0)
                self.assertEqual((c.shape[0], c.shape[1]), (rows, cols))
                self.assertEqual((h.shape[0], h.shape[1]), (rows, cols))
                self.assertEqual((i.shape[0], i.shape[1]), (rows, cols))
                self.assertEqual((l.shape[0], l.shape[1]), (rows, cols))
                self.assertEqual((f.shape[0], f.shape[1]), (rows, cols))
                self.assertEqual((d.shape[0], d.shape[1]), (rows, cols))
