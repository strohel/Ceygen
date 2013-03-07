/* Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
 * Distributed under the terms of the GNU General Public License v2 or any
 * later version of the license, at your option. */

#ifndef DISPATCH_H
#define DISPATCH_H
#include "eigen_cpp.h"


// dummy compile time-only classes to differentiate between various contiguity types
struct CContig { enum {
	Layout = RowMajor,
	InnerStride = 1
};};
struct FContig { enum {
	Layout = ColMajor,
	InnerStride = 1
};};
struct NContig { enum {
	Layout = RowMajor,
	InnerStride = Dynamic
};};

// function taking memview, scalar arguments
#define as_func(name, XContigType) \
void (*name)(dtype *, Py_ssize_t *, Py_ssize_t *, XContigType, dtype *)

// function taking memview, memoryview arguments
#define aa_func(name, XContigType, OContigType) \
void (*name)(dtype *, Py_ssize_t *, Py_ssize_t *, XContigType, dtype *, Py_ssize_t *, Py_ssize_t *, OContigType)

// function taking memview, memview, scalar pointer arguments
#define aas_func(name, XContigType, YContigType) \
void (*name)(dtype *, Py_ssize_t *, Py_ssize_t *, XContigType, dtype *, Py_ssize_t *, Py_ssize_t *, YContigType, dtype *)

// function taking three memview arguments
#define aaa_func(name, XContigType, YContigType, OContigType) \
void (*name)(dtype *, Py_ssize_t *, Py_ssize_t *, XContigType, dtype *, Py_ssize_t *, Py_ssize_t *, YContigType, dtype *, Py_ssize_t *, Py_ssize_t *, OContigType)

template<typename dtype>
struct VSDispatcher
{
	union VSFuncs {
		as_func(c, CContig); as_func(n, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *o, as_func(c, CContig), as_func(n, NContig))
	{
		/* it is better to just store function pointer inside the if/else branches,
		 * because calling a function with a lot of parameters generates a lot of
		 * (conditional) code which then causes cache misses */
		VSFuncs tocall;
		if(x_strides[0] == sizeof(dtype))
			tocall.c = c;
		else
			tocall.n = n;
		/* Following is a vile hack! We pretend to call the nns variant, alhough the
		 * pointer may point to any of the variants stored in the union. This works because
		 * the functions only differ in CContig/FContig/NContig arguments, and all these
		 * are empty structures, which must have same (no) memory representation, thus
		 * all the functions must have the same call signature. Another alternative would
		 * be to cast function pointer, but abusing union was deemed slightly less ugly. */
		tocall.n(x_data, x_shape, x_strides, NContig(), o);
	}
};

template<typename dtype>
struct VVSDispatcher
{
	union VVSFuncs {
		aas_func(ccs, CContig, CContig); aas_func(cns, CContig, NContig);
		aas_func(ncs, NContig, CContig); aas_func(nns, NContig, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
			dtype *o,
			aas_func(ccs, CContig, CContig), aas_func(cns, CContig, NContig),
			aas_func(ncs, NContig, CContig), aas_func(nns, NContig, NContig))
	{
		VVSFuncs tocall;
		if(x_strides[0] == sizeof(dtype)) {
			if(y_strides[0] == sizeof(dtype))
				tocall.ccs = ccs;
			else
				tocall.cns = cns;
		} else {
			if(y_strides[0] == sizeof(dtype))
				tocall.ncs = ncs;
			else
				tocall.nns = nns;
		}
		tocall.nns(x_data, x_shape, x_strides, NContig(), y_data, y_shape, y_strides, NContig(), o);
	}
};

template<typename dtype>
struct VVVDispatcher
{
	union VVVFuncs {
		aaa_func(ccc, CContig, CContig, CContig); aaa_func(ccn, CContig, CContig, NContig);
		aaa_func(cnc, CContig, NContig, CContig); aaa_func(cnn, CContig, NContig, NContig);
		aaa_func(ncc, NContig, CContig, CContig); aaa_func(ncn, NContig, CContig, NContig);
		aaa_func(nnc, NContig, NContig, CContig); aaa_func(nnn, NContig, NContig, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
			dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
			aaa_func(ccc, CContig, CContig, CContig), aaa_func(ccn, CContig, CContig, NContig),
			aaa_func(cnc, CContig, NContig, CContig), aaa_func(cnn, CContig, NContig, NContig),
			aaa_func(ncc, NContig, CContig, CContig), aaa_func(ncn, NContig, CContig, NContig),
			aaa_func(nnc, NContig, NContig, CContig), aaa_func(nnn, NContig, NContig, NContig))
	{
		VVVFuncs tocall;
		if(x_strides[0] == sizeof(dtype)) {
			if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[0] == sizeof(dtype))
					tocall.ccc = ccc;
				else
					tocall.ccn = ccn;
			} else {
				if(o_strides[0] == sizeof(dtype))
					tocall.cnc = cnc;
				else
					tocall.cnn = cnn;
			}
		} else {
			if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[0] == sizeof(dtype))
					tocall.ncc = ncc;
				else
					tocall.ncn = ncn;
			} else {
				if(o_strides[0] == sizeof(dtype))
					tocall.nnc = nnc;
				else
					tocall.nnn = nnn;
			}
		}
		tocall.nnn(x_data, x_shape, x_strides, NContig(), y_data, y_shape, y_strides, NContig(), o_data, o_shape, o_strides, NContig());
	}
};

template<typename dtype>
struct MSDispatcher
{
	union MSFuncs {
		as_func(c, CContig); as_func(f, FContig); as_func(n, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *o,
			as_func(c, CContig), as_func(f, FContig), as_func(n, NContig))
	{
		MSFuncs tocall;
		if(x_strides[1] == sizeof(dtype))
			tocall.c = c;
		else if(x_strides[0] == sizeof(dtype))
			tocall.f = f;
		else
			tocall.n = n;
		tocall.n(x_data, x_shape, x_strides, NContig(), o);
	}
};

template<typename dtype>
struct MVDispatcher
{
	union MVFuncs {
		aa_func(cc, CContig, CContig); aa_func(cn, CContig, NContig);
		aa_func(fc, FContig, CContig); aa_func(fn, FContig, NContig);
		aa_func(nc, NContig, CContig); aa_func(nn, NContig, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
			aa_func(cc, CContig, CContig), aa_func(cn, CContig, NContig),
			aa_func(fc, FContig, CContig), aa_func(fn, FContig, NContig),
			aa_func(nc, NContig, CContig), aa_func(nn, NContig, NContig))
	{
		MVFuncs tocall;
		if(x_strides[1] == sizeof(dtype)) {
			if(o_strides[0] == sizeof(dtype))
				tocall.cc = cc;
			else
				tocall.cn = cn;
		} else if(x_strides[0] == sizeof(dtype)) {
			if(o_strides[0] == sizeof(dtype))
				tocall.fc = fc;
			else
				tocall.fn = fn;
		} else {
			if(o_strides[0] == sizeof(dtype))
				tocall.nc = nc;
			else
				tocall.nn = nn;
		}
		tocall.nn(x_data, x_shape, x_strides, NContig(), o_data, o_shape, o_strides, NContig());
	}
};

template<typename dtype>
struct MMSDispatcher
{
	union MMSFuncs {
		aas_func(ccs, CContig, CContig); aas_func(cfs, CContig, FContig); aas_func(cns, CContig, NContig);
		aas_func(fcs, FContig, CContig); aas_func(ffs, FContig, FContig); aas_func(fns, FContig, NContig);
		aas_func(ncs, NContig, CContig); aas_func(nfs, NContig, FContig); aas_func(nns, NContig, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
			dtype *o,
			aas_func(ccs, CContig, CContig), aas_func(cfs, CContig, FContig), aas_func(cns, CContig, NContig),
			aas_func(fcs, FContig, CContig), aas_func(ffs, FContig, FContig), aas_func(fns, FContig, NContig),
			aas_func(ncs, NContig, CContig), aas_func(nfs, NContig, FContig), aas_func(nns, NContig, NContig))
	{
		MMSFuncs tocall;
		if(x_strides[1] == sizeof(dtype)) {
			if(y_strides[1] == sizeof(dtype))
				tocall.ccs = ccs;
			else if(y_strides[0] == sizeof(dtype))
				tocall.cfs = cfs;
			else
				tocall.cns = cns;
		} else if(x_strides[0] == sizeof(dtype)) {
			if(y_strides[1] == sizeof(dtype))
				tocall.fcs = fcs;
			else if(y_strides[0] == sizeof(dtype))
				tocall.ffs = ffs;
			else
				tocall.fns = fns;
		} else {
			if(y_strides[1] == sizeof(dtype))
				tocall.ncs = ncs;
			else if(y_strides[0] == sizeof(dtype))
				tocall.nfs = nfs;
			else
				tocall.nns = nns;
		}
		tocall.nns(x_data, x_shape, x_strides, NContig(), y_data, y_shape, y_strides, NContig(), o);
	}
};

template<typename dtype>
struct MVVDispatcher
{
	union MVVFuncs {
		aaa_func(ccc, CContig, CContig, CContig); aaa_func(ccn, CContig, CContig, NContig);
		aaa_func(cnc, CContig, NContig, CContig); aaa_func(cnn, CContig, NContig, NContig);
		aaa_func(fcc, FContig, CContig, CContig); aaa_func(fcn, FContig, CContig, NContig);
		aaa_func(fnc, FContig, NContig, CContig); aaa_func(fnn, FContig, NContig, NContig);
		aaa_func(ncc, NContig, CContig, CContig); aaa_func(ncn, NContig, CContig, NContig);
		aaa_func(nnc, NContig, NContig, CContig); aaa_func(nnn, NContig, NContig, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
			dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
			aaa_func(ccc, CContig, CContig, CContig), aaa_func(ccn, CContig, CContig, NContig),
			aaa_func(cnc, CContig, NContig, CContig), aaa_func(cnn, CContig, NContig, NContig),
			aaa_func(fcc, FContig, CContig, CContig), aaa_func(fcn, FContig, CContig, NContig),
			aaa_func(fnc, FContig, NContig, CContig), aaa_func(fnn, FContig, NContig, NContig),
			aaa_func(ncc, NContig, CContig, CContig), aaa_func(ncn, NContig, CContig, NContig),
			aaa_func(nnc, NContig, NContig, CContig), aaa_func(nnn, NContig, NContig, NContig))
	{
		MVVFuncs tocall;
		if(x_strides[1] == sizeof(dtype)) {
			if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[0] == sizeof(dtype))
					tocall.ccc = ccc;
				else
					tocall.ccn = ccn;
			} else {
				if(o_strides[0] == sizeof(dtype))
					tocall.cnc = cnc;
				else
					tocall.cnn = cnn;
			}
		} else if(x_strides[0] == sizeof(dtype)) {
			if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[0] == sizeof(dtype))
					tocall.fcc = fcc;
				else
					tocall.fcn = fcn;
			} else {
				if(o_strides[0] == sizeof(dtype))
					tocall.fnc = fnc;
				else
					tocall.fnn = fnn;
			}
		} else {
			if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[0] == sizeof(dtype))
					tocall.ncc = ncc;
				else
					tocall.ncn = ncn;
			} else {
				if(o_strides[0] == sizeof(dtype))
					tocall.nnc = nnc;
				else
					tocall.nnn = nnn;
			}
		}
		tocall.nnn(x_data, x_shape, x_strides, NContig(), y_data, y_shape, y_strides, NContig(), o_data, o_shape, o_strides, NContig());
	}
};

template<typename dtype>
struct MMMDispatcher
{
	union MMMFuncs {
		aaa_func(ccc, CContig, CContig, CContig); aaa_func(ccf, CContig, CContig, FContig); aaa_func(ccn, CContig, CContig, NContig);
		aaa_func(cfc, CContig, FContig, CContig); aaa_func(cff, CContig, FContig, FContig); aaa_func(cfn, CContig, FContig, NContig);
		aaa_func(cnc, CContig, NContig, CContig); aaa_func(cnf, CContig, NContig, FContig); aaa_func(cnn, CContig, NContig, NContig);
		aaa_func(fcc, FContig, CContig, CContig); aaa_func(fcf, FContig, CContig, FContig); aaa_func(fcn, FContig, CContig, NContig);
		aaa_func(ffc, FContig, FContig, CContig); aaa_func(fff, FContig, FContig, FContig); aaa_func(ffn, FContig, FContig, NContig);
		aaa_func(fnc, FContig, NContig, CContig); aaa_func(fnf, FContig, NContig, FContig); aaa_func(fnn, FContig, NContig, NContig);
		aaa_func(ncc, NContig, CContig, CContig); aaa_func(ncf, NContig, CContig, FContig); aaa_func(ncn, NContig, CContig, NContig);
		aaa_func(nfc, NContig, FContig, CContig); aaa_func(nff, NContig, FContig, FContig); aaa_func(nfn, NContig, FContig, NContig);
		aaa_func(nnc, NContig, NContig, CContig); aaa_func(nnf, NContig, NContig, FContig); aaa_func(nnn, NContig, NContig, NContig);
	};

	inline void run(dtype *x_data, Py_ssize_t *x_shape, Py_ssize_t *x_strides,
			dtype *y_data, Py_ssize_t *y_shape, Py_ssize_t *y_strides,
			dtype *o_data, Py_ssize_t *o_shape, Py_ssize_t *o_strides,
			aaa_func(ccc, CContig, CContig, CContig), aaa_func(ccf, CContig, CContig, FContig), aaa_func(ccn, CContig, CContig, NContig),
			aaa_func(cfc, CContig, FContig, CContig), aaa_func(cff, CContig, FContig, FContig), aaa_func(cfn, CContig, FContig, NContig),
			aaa_func(cnc, CContig, NContig, CContig), aaa_func(cnf, CContig, NContig, FContig), aaa_func(cnn, CContig, NContig, NContig),
			aaa_func(fcc, FContig, CContig, CContig), aaa_func(fcf, FContig, CContig, FContig), aaa_func(fcn, FContig, CContig, NContig),
			aaa_func(ffc, FContig, FContig, CContig), aaa_func(fff, FContig, FContig, FContig), aaa_func(ffn, FContig, FContig, NContig),
			aaa_func(fnc, FContig, NContig, CContig), aaa_func(fnf, FContig, NContig, FContig), aaa_func(fnn, FContig, NContig, NContig),
			aaa_func(ncc, NContig, CContig, CContig), aaa_func(ncf, NContig, CContig, FContig), aaa_func(ncn, NContig, CContig, NContig),
			aaa_func(nfc, NContig, FContig, CContig), aaa_func(nff, NContig, FContig, FContig), aaa_func(nfn, NContig, FContig, NContig),
			aaa_func(nnc, NContig, NContig, CContig), aaa_func(nnf, NContig, NContig, FContig), aaa_func(nnn, NContig, NContig, NContig))
	{
		MMMFuncs tocall;
		if(x_strides[1] == sizeof(dtype)) {
			if(y_strides[1] == sizeof(dtype)) {
				if(o_strides[1] == sizeof(dtype))
					tocall.ccc = ccc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.ccf = ccf;
				else
					tocall.ccn = ccn;
			} else if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[1] == sizeof(dtype))
					tocall.cfc = cfc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.cff = cff;
				else
					tocall.cfn = cfn;
			} else {
				if(o_strides[1] == sizeof(dtype))
					tocall.cnc = cnc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.cnf = cnf;
				else
					tocall.cnn = cnn;
			}
		} else if(x_strides[0] == sizeof(dtype)) {
			if(y_strides[1] == sizeof(dtype)) {
				if(o_strides[1] == sizeof(dtype))
					tocall.fcc = fcc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.fcf = fcf;
				else
					tocall.fcn = fcn;
			} else if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[1] == sizeof(dtype))
					tocall.ffc = ffc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.fff = fff;
				else
					tocall.ffn = ffn;
			} else {
				if(o_strides[1] == sizeof(dtype))
					tocall.fnc = fnc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.fnf = fnf;
				else
					tocall.fnn = fnn;
			}
		} else {
			if(y_strides[1] == sizeof(dtype)) {
				if(o_strides[1] == sizeof(dtype))
					tocall.ncc = ncc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.ncf = ncf;
				else
					tocall.ncn = ncn;
			} else if(y_strides[0] == sizeof(dtype)) {
				if(o_strides[1] == sizeof(dtype))
					tocall.nfc = nfc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.nff = nff;
				else
					tocall.nfn = nfn;
			} else {
				if(o_strides[1] == sizeof(dtype))
					tocall.nnc = nnc;
				else if(o_strides[0] == sizeof(dtype))
					tocall.nnf = nnf;
				else
					tocall.nnn = nnn;
			}
		}
		tocall.nnn(x_data, x_shape, x_strides, NContig(), y_data, y_shape, y_strides, NContig(), o_data, o_shape, o_strides, NContig());
	}
};

#endif // DISPATCH_H
