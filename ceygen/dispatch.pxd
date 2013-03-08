# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

cdef extern from "dispatch.h":
    # dummy classes to differentiate between various contiguity types
    cdef cppclass CContig:
        pass
    cdef cppclass FContig:
        pass
    cdef cppclass NContig:
        pass

    cdef cppclass VSDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *o,
                 void (*c)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*n)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
        ) nogil except +

    cdef cppclass VVSDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
                 Scalar *o,
                 void (*ccs)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*cns)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
                 void (*ncs)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*nns)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
        ) nogil except +

    cdef cppclass VVVDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
                 Scalar *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
                 void (*ccc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*ccn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*cnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*cnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*ncc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*ncn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*nnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*nnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
        ) nogil except +

    cdef cppclass MSDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *o,
                 void (*c)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*f)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *) nogil,
                 void (*n)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
        ) nogil except +

    cdef cppclass MVDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
                 void (*cc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*cn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*fc)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*fn)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*nc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*nn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
        ) nogil except +

    cdef cppclass MMSDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
                 Scalar *o,
                 void (*ccs)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*cfs)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *) nogil,
                 void (*cns)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
                 void (*fcs)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*ffs)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *) nogil,
                 void (*fns)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
                 void (*ncs)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *) nogil,
                 void (*nfs)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *) nogil,
                 void (*nns)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *) nogil,
        ) nogil except +

    cdef cppclass MVVDispatcher[Scalar]:
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
                 Scalar *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
                 void (*ccc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*ccn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*cnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*cnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*fcc)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*fcn)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*fnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*fnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*ncc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*ncn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*nnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*nnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
        ) nogil except +

    cdef cppclass MMMDispatcher[Scalar]:
        # okay, this is ridiculous, but this function is inlined and written only once...
        void run(Scalar *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
                 Scalar *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
                 Scalar *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
                 void (*ccc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*ccf)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*ccn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*cfc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*cff)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*cfn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*cnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*cnf)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*cnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*fcc)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*fcf)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*fcn)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*ffc)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*fff)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*ffn)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*fnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*fnf)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*fnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*ncc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*ncf)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*ncn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*nfc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*nff)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*nfn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
                 void (*nnc)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, CContig) nogil,
                 void (*nnf)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, FContig) nogil,
                 void (*nnn)(Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig, Scalar *, Py_ssize_t *, Py_ssize_t *, NContig) nogil,
        ) nogil except +


ctypedef fused XVectorContiguity:
    CContig
    NContig

ctypedef fused YVectorContiguity:
    CContig
    NContig

ctypedef fused OVectorContiguity:
    CContig
    NContig

ctypedef fused XMatrixContiguity:
    CContig
    FContig
    NContig

ctypedef fused YMatrixContiguity:
    CContig
    FContig
    NContig

ctypedef fused OMatrixContiguity:
    CContig
    FContig
    NContig
