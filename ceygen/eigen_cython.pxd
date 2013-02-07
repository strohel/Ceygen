# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

# note: Cython doesn't like VectorMap[Scalar] anywhere except in cdef cppclass ...
# declaration. Using just the class without template param everywhere else works
# well

from libcpp cimport bool

from dtype cimport dtype


cdef extern from "eigen_cpp.h":
    void c_set_is_malloc_allowed "internal::set_is_malloc_allowed"(bool) nogil

    cdef cppclass BaseMap[Scalar]:
        # "constructor":
        void init(Scalar *, const Py_ssize_t *, const Py_ssize_t *) nogil except +

        # our own methods:
        void assign(BaseMap) nogil except +
        void assign_inverse(BaseMap) nogil except +
        void noalias_assign(BaseMap) nogil except +
        void noalias_assign_dot_mm "noalias_assign_dot_mm<NoContig, NoContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        # following are different template specializations, Ceygen unfortunately doesn't support yet function templates
        void noalias_assign_dot_cc "noalias_assign_dot_mm<CContig, CContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        void noalias_assign_dot_cf "noalias_assign_dot_mm<CContig, FContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        void noalias_assign_dot_fc "noalias_assign_dot_mm<FContig, CContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +
        void noalias_assign_dot_ff "noalias_assign_dot_mm<FContig, FContig>"(
            Scalar *x_data, const Py_ssize_t *x_shape, const Py_ssize_t *x_strides,
            Scalar *y_data, const Py_ssize_t *y_shape, const Py_ssize_t *y_strides) nogil except +

        # exported Eigen methods
        Scalar determinant() nogil except +
        Scalar dot(BaseMap) nogil except +

        # this is a huge cheat, these operators don't map 1:1 to actual C++ operators at
        # all; but the declarations here are just to tell that the operators are possible..
        BaseMap operator+(BaseMap) nogil except +
        BaseMap operator-(BaseMap) nogil except +
        BaseMap operator*(BaseMap) nogil except +
        BaseMap operator/(BaseMap) nogil except +

    cdef cppclass VectorMap[Scalar](BaseMap):
        pass

    cdef cppclass RowVectorMap[Scalar](BaseMap):
        pass

    cdef cppclass Array1DMap[Scalar](BaseMap):
        # must be here, Cython has problems inheriting overloads, http://trac.cython.org/cython_trac/ticket/800
        BaseMap operator+(Scalar) nogil except +
        BaseMap operator*(Scalar) nogil except +

    cdef cppclass MatrixMap[Scalar](BaseMap):
        pass

    cdef cppclass MatrixMapTODO[Scalar, ContiguityType](BaseMap):
        pass

    cdef cppclass Array2DMap[Scalar](BaseMap):
        # http://trac.cython.org/cython_trac/ticket/800
        BaseMap operator+(Scalar) nogil except +
        BaseMap operator*(Scalar) nogil except +

    # dummy classes to differentiate between various contiguity types
    cdef cppclass CContig:
        pass
    cdef cppclass FContig:
        pass
    cdef cppclass NContig:
        pass


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

cdef inline bint dispatch_mmm(
        dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
        dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
        dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
        bint (*ccc)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*ccf)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*ccn)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*cfc)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*cff)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*cfn)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*cnc)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*cnf)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*cnn)(dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*fcc)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*fcf)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*fcn)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*ffc)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*fff)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*ffn)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*fnc)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*fnf)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*fnn)(dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*ncc)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*ncf)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*ncn)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*nfc)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*nff)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*nfn)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        bint (*nnc)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, CContig) nogil except False,
        bint (*nnf)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, FContig) nogil except False,
        bint (*nnn)(dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig, dtype *, Py_ssize_t *, Py_ssize_t *, NContig) nogil except False,
        ) nogil except -1:
    # okay, this is ridiculous, but this function is inlined and written only once...
    # workaround for Cython crash when NoContig() etc is used in-place as argument:
    cdef NContig ncontig
    cdef CContig ccontig
    cdef FContig fcontig
    if x_strides[1] == sizeof(dtype):
        if y_strides[1] == sizeof(dtype):
            if o_strides[1] == sizeof(dtype):
                return ccc(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return ccf(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, fcontig)
            else:
                return ccn(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, ncontig)
        elif y_strides[0] == sizeof(dtype):
            if o_strides[1] == sizeof(dtype):
                return cfc(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return cff(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, fcontig)
            else:
                return cfn(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, ncontig)
        else:
            if o_strides[1] == sizeof(dtype):
                return cnc(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return cnf(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, fcontig)
            else:
                return cnn(x_data, x_shape, x_strides, ccontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, ncontig)
    elif x_strides[0] == sizeof(dtype):
        if y_strides[1] == sizeof(dtype):
            if o_strides[1] == sizeof(dtype):
                return fcc(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return fcf(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, fcontig)
            else:
                return fcn(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, ncontig)
        elif y_strides[0] == sizeof(dtype):
            if o_strides[1] == sizeof(dtype):
                return ffc(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return fff(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, fcontig)
            else:
                return ffn(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, ncontig)
        else:
            if o_strides[1] == sizeof(dtype):
                return fnc(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return fnf(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, fcontig)
            else:
                return fnn(x_data, x_shape, x_strides, fcontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, ncontig)
    else:
        if y_strides[1] == sizeof(dtype):
            if o_strides[1] == sizeof(dtype):
                return ncc(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return ncf(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, fcontig)
            else:
                return ncn(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, ccontig, o_data, o_shape, o_strides, ncontig)
        elif y_strides[0] == sizeof(dtype):
            if o_strides[1] == sizeof(dtype):
                return nfc(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return nff(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, fcontig)
            else:
                return nfn(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, fcontig, o_data, o_shape, o_strides, ncontig)
        else:
            if o_strides[1] == sizeof(dtype):
                return nnc(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, ccontig)
            elif o_strides[0] == sizeof(dtype):
                return nnf(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, fcontig)
            else:
                return nnn(x_data, x_shape, x_strides, ncontig, y_data, y_shape, y_strides, ncontig, o_data, o_shape, o_strides, ncontig)
