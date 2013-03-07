# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

import numpy as np

from support import CeygenTestCase
from ceygen.dispatch cimport *


globalstatus = ''

cdef void as_func(
        double *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        double *o) with gil:
    global globalstatus
    globalstatus = ''
    if XMatrixContiguity is CContig:
        globalstatus += 'C'
    if XMatrixContiguity is FContig:
        globalstatus += 'F'
    if XMatrixContiguity is NContig:
        globalstatus += 'N'

cdef void aa_func(
        double *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        double *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy) with gil:
    global globalstatus
    globalstatus = ''
    if XMatrixContiguity is CContig:
        globalstatus += 'C'
    if XMatrixContiguity is FContig:
        globalstatus += 'F'
    if XMatrixContiguity is NContig:
        globalstatus += 'N'

    if YMatrixContiguity is CContig:
        globalstatus += 'C'
    if YMatrixContiguity is FContig:
        globalstatus += 'F'
    if YMatrixContiguity is NContig:
        globalstatus += 'N'

cdef void aas_func(
        double *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        double *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        double *o) with gil:
    global globalstatus
    globalstatus = ''
    if XMatrixContiguity is CContig:
        globalstatus += 'C'
    if XMatrixContiguity is FContig:
        globalstatus += 'F'
    if XMatrixContiguity is NContig:
        globalstatus += 'N'

    if YMatrixContiguity is CContig:
        globalstatus += 'C'
    if YMatrixContiguity is FContig:
        globalstatus += 'F'
    if YMatrixContiguity is NContig:
        globalstatus += 'N'

cdef void aaa_func(
        double *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides, XMatrixContiguity x_dummy,
        double *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides, YMatrixContiguity y_dummy,
        double *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides, OMatrixContiguity o_dummy) with gil:
    global globalstatus
    globalstatus = ''
    if XMatrixContiguity is CContig:
        globalstatus += 'C'
    if XMatrixContiguity is FContig:
        globalstatus += 'F'
    if XMatrixContiguity is NContig:
        globalstatus += 'N'

    if YMatrixContiguity is CContig:
        globalstatus += 'C'
    if YMatrixContiguity is FContig:
        globalstatus += 'F'
    if YMatrixContiguity is NContig:
        globalstatus += 'N'

    if OMatrixContiguity is CContig:
        globalstatus += 'C'
    if OMatrixContiguity is FContig:
        globalstatus += 'F'
    if OMatrixContiguity is NContig:
        globalstatus += 'N'

v_ccontig = (np.array([1., 2.]), 'C')
v_ncontig = (np.array([[1., 2.], [3., 4.]])[:, 1], 'N')

m_ccontig = (np.array([[1., 2.], [3., 4.]]), 'C')
m_fcontig = (np.array([[1., 2.], [3., 4.]], order='F'), 'F')
m_ncontig = (np.array([[[1., 2.], [3., 4.]],
                       [[5., 6.], [7., 8.]]])[:, :, 1], 'N')

class TestDispatch(CeygenTestCase):

    def test_vs(self):
        cdef VSDispatcher[double] dispatcher
        cdef double[:] x
        for X in (v_ccontig, v_ncontig):
                x = X[0]
                dispatcher.run(&x[0], x.shape, x.strides, <double *> 0,
                        as_func, as_func)
                self.assertEquals(globalstatus, X[1])

    def test_vvs(self):
        cdef VVSDispatcher[double] dispatcher
        cdef double[:] x, y
        for X in (v_ccontig, v_ncontig):
            for Y in (v_ccontig, v_ncontig):
                x, y = X[0], Y[0]
                dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides, <double *> 0,
                        aas_func, aas_func, aas_func, aas_func)
                self.assertEquals(globalstatus, X[1] + Y[1])

    def test_vvv(self):
        cdef VVVDispatcher[double] dispatcher
        cdef double[:] x, y, z
        for X in (v_ccontig, v_ncontig):
            for Y in (v_ccontig, v_ncontig):
                for Z in (v_ccontig, v_ncontig):
                    x, y, z = X[0], Y[0], Z[0]
                    dispatcher.run(&x[0], x.shape, x.strides, &y[0], y.shape, y.strides,
                            &z[0], z.shape, z.strides, aaa_func, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func, aaa_func)
                    self.assertEquals(globalstatus, X[1] + Y[1] + Z[1])

    def test_ms(self):
        cdef MSDispatcher[double] dispatcher
        cdef double[:, :] x
        for X in (m_ccontig, m_fcontig, m_ncontig):
                x = X[0]
                dispatcher.run(&x[0, 0], x.shape, x.strides, <double *> 0,
                        as_func, as_func, as_func)
                self.assertEquals(globalstatus, X[1])

    def test_mv(self):
        cdef MVDispatcher[double] dispatcher
        cdef double[:, :] x
        cdef double[:] y
        for X in (m_ccontig, m_fcontig, m_ncontig):
            for Y in (v_ccontig, v_ncontig):
                x, y = X[0], Y[0]
                dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0], y.shape, y.strides,
                        aa_func, aa_func, aa_func, aa_func, aa_func, aa_func)
                self.assertEquals(globalstatus, X[1] + Y[1])

    def test_mms(self):
        cdef MMSDispatcher[double] dispatcher
        cdef double[:, :] x, y
        for X in (m_ccontig, m_fcontig, m_ncontig):
            for Y in (m_ccontig, m_fcontig, m_ncontig):
                x, y = X[0], Y[0]
                dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides, <double *> 0,
                        aas_func, aas_func, aas_func, aas_func, aas_func, aas_func,
                        aas_func, aas_func, aas_func)
                self.assertEquals(globalstatus, X[1] + Y[1])

    def test_mvv(self):
        cdef MVVDispatcher[double] dispatcher
        cdef double[:, :] x
        cdef double[:] y, z
        for X in (m_ccontig, m_fcontig, m_ncontig):
            for Y in (v_ccontig, v_ncontig):
                for Z in (v_ccontig, v_ncontig):
                    x, y, z = X[0], Y[0], Z[0]
                    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0], y.shape, y.strides,
                            &z[0], z.shape, z.strides, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func)
                    self.assertEquals(globalstatus, X[1] + Y[1] + Z[1])

    def test_mmm(self):
        cdef MMMDispatcher[double] dispatcher
        cdef double[:, :] x, y, z
        for X in (m_ccontig, m_fcontig, m_ncontig):
            for Y in (m_ccontig, m_fcontig, m_ncontig):
                for Z in (m_ccontig, m_fcontig, m_ncontig):
                    x, y, z = X[0], Y[0], Z[0]
                    dispatcher.run(&x[0, 0], x.shape, x.strides, &y[0, 0], y.shape, y.strides,
                            &z[0, 0], z.shape, z.strides, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func, aaa_func, aaa_func, aaa_func,
                            aaa_func, aaa_func, aaa_func, aaa_func, aaa_func, aaa_func)
                    self.assertEquals(globalstatus, X[1] + Y[1] + Z[1])
