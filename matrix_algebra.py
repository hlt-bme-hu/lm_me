#coding: utf8

from __future__ import print_function

import numpy
import sys
import os.path
import argparse
from collections import defaultdict
from itertools import chain
import theano
import theano.tensor as T
from auxiliary import forward

class AlgebraException(Exception):
    pass

def stack(tensors, library, axis):
    stacked = library.stack(tensors, axis)
    if library != numpy:
        return library.unbroadcast(stacked, axis)
    else:
        return stacked

class MatrixAlgebra(object):
    def __init__(self):
        self.positions = []
        self.param = 0
        self.dim = 0

    def to_array(self, x, dtype="float64"):
        position_dic = get_dict(self.positions)
        if len(x.shape) == 2:
            result = numpy.zeros((x.shape[0], self.dim, self.dim), dtype=dtype)
            for i, p in position_dic.items():
                result[:, p[0], p[1]] = x[:, i]
        else:
            result = numpy.zeros((self.dim, self.dim), dtype=dtype)
            for i, p in position_dic.items():
                result[p] = x[i]
        return result
    
    def renormalize_forward(self, M, cutoff=float("-inf"), library=numpy, copy=True):
        expM = forward(M, cutoff, library=library, copy=copy)
        if copy:
            return expM / self.b(expM, library)[None, :]
        else:
            # M is already updated
            M /= self.b(M, library)[None, :]
            
    def renormalize_forward2(self, M, cutoff=float("-inf"), library=numpy, copy=True):
        expM = forward(M, cutoff, library=library, copy=copy)
        if copy:            
            return expM / self.b2(expM, library)
        else:
            # M is already updated
            M /= self.b2(M, library)
 
    def renormalize(self, M, cutoff=float("-inf"), library=numpy, copy=True):
        bb = library.log2(self.b(forward(M, cutoff, library), library))
        if copy:
            return M - bb[None, :]
        else:
            M -= bb[None, :]

    def renormalize2(self, M, cutoff=float("-inf"), library=numpy, copy=True):
        bb = library.log2(self.b2(forward(M, cutoff, library), library))
        if copy:
            return M - bb
        else:
            M -= bb

    def me_layer_T(self, M, indices):
        """
        (time * param * batch)
        """
        batch_size = indices.shape[0]
        start_left = T.unbroadcast(T.ones((batch_size, 1)).dot(self.identity), 0, 1)
        lefts, updates = theano.scan(
            lambda s, l, M_: self.batch_dot_T(l, M_[s, :].transpose(), T),
            sequences=indices.transpose(),
            outputs_info=start_left.transpose(),
            non_sequences=[M], strict=True)
        return lefts, updates

    def me_layer(self, M, indices):
        """
        (time * batch * param)
        """
        batch_size = indices.shape[0]
        start_left = T.unbroadcast(T.ones((batch_size, 1)).dot(self.identity), 0, 1)
        lefts, updates = theano.scan(
            lambda s, l, M_: self.batch_dot(l, M_[s, :], T),
            sequences=indices.transpose(),
            outputs_info=start_left,
            non_sequences=[M], strict=True)
        return lefts, updates
    
    def right_stochastic_layer(self, M, indices):
        """
        computes a right-stochastic embedding layer
        returns a (time * batch * dim) array
        """
        start_left = T.ones((indices.shape[0], self.dim))
        return theano.scan(
            lambda s, l, M_: self.right_batch_dot(l, M_[s, :], T),
            sequences=indices.transpose(),
            outputs_info=start_left, non_sequences=[M], strict=True)

class CommutativeAlgebra(MatrixAlgebra):
    def __init__(self, dim):
        self.param = dim
        self.dim = dim
        self.positions = [(i, i) for i in range(dim)]
        self.identity = numpy.ones((1, dim), dtype=theano.config.floatX)

    def b(self, M, library=numpy):
        return M.sum(0)

    def b2(self, M, library=numpy):
        return M
        
    def batch_dot(self, X, Y, library=numpy):
        return X*Y
    
    def batch_dot_T(self, X, Y, library=numpy):
        return X*Y
        
    def right_batch_dot(self, X, Y, library=numpy):
        return X*Y
        
class DenseAlgebra(MatrixAlgebra):
    def __init__(self, dim):
        self.param = dim*dim
        self.dim = dim
        self.positions = [(i, j) for i in range(dim) for j in range(dim)]
        self.identity = numpy.eye(dim, dtype=theano.config.floatX).reshape((1, dim**2))

    def b(self, M, library=numpy):
        bb = self.to_mtx(M).sum(0).sum(1)
        return (bb[:, None]*library.ones(self.dim)[None, :]).reshape((self.dim**2,))

    def b2(self, M, library=numpy):
        bb = self.to_mtx(M).sum(2)
        return (bb[:, :, None]*library.ones(self.dim)[None, None, :]).reshape((-1,self.dim**2))
    
    def to_mtx(self, x):
        return x.reshape((x.shape[0], self.dim, self.dim))

    def to_vec(self, x):
        return x.reshape((x.shape[0], self.dim**2))

    def batch_dot(self, X, Y, library=numpy):
        dot_f = numpy.matmul if library == numpy else T.batched_dot
        return self.to_vec(dot_f(self.to_mtx(X), self.to_mtx(Y)))

    def me_layer(self, M, indices):
        batch_size = indices.shape[0]
        
        start_left = self.to_mtx(T.ones((batch_size, 1)).dot(self.identity))
        lefts, updates = theano.scan(
            lambda s, l, M_: T.batched_dot(l, self.to_mtx(M_[s, :])),
            sequences=indices.transpose(),
            outputs_info=start_left,
            non_sequences=[M], strict=True)
        return lefts.reshape((-1, batch_size, self.dim**2)), updates

    def right_batch_dot(self, X, Y, library=numpy):
        """
        performs vector-matrix batched dot product
        X is a (batch * dim) array
        Y is a (batch * param) array
        output is a (batch * dim) array
        """
        dot_f = numpy.matmul if library == numpy else T.batched_dot
        M = dot_f(X.reshape((-1, 1, self.dim)), self.to_mtx(Y))
        return M.reshape((-1, self.dim))

class SparseAlgebra(MatrixAlgebra):
    def __init__(self, positions, check=True):
        self.positions = positions
        self.position_dic = get_dict(positions)
        self.param = len(positions)
        self.dim = get_dim(positions)
        self.kernel = get_kernel(self.position_dic)

        self.right_kernel = [[k] + list(self.position_dic[k]) for k in self.position_dic]
        self.right_kernel = sorted(self.right_kernel, key=lambda x: x[1:])
        self.right_kernel = numpy.array(self.right_kernel, dtype=numpy.int32)
        self.right_kernel_shr = theano.shared(self.right_kernel)
        
        if check and not check_positions(positions):
            raise AlgebraException()
        
        self.identity = numpy.zeros((1, self.param), dtype=theano.config.floatX)
        for i, p in self.position_dic.items():
            if p[0] == p[1]:
                self.identity[0, i] = 1

    def batch_dot_T(self, X, Y, library=numpy):
        """
        performs sparse batched dot product
        inputs and the output are all 'param * batch' size
        """
        kernel = get_kernel(get_dict(self.positions))
        tensors = [sum(X[i] * Y[j] for i, j in kernel[k]) for k in range(self.param)]
        return stack(tensors, library, 0)

    def batch_dot(self, X, Y, library=numpy):
        """
        performs sparse batched dot product
        inputs and the output are all 'batch * param' size
        """
        kernel = get_kernel(get_dict(self.positions))
        tensors = [sum(X[:, i] * Y[:, j] for i, j in kernel[k]) for k in range(self.param)]
        return stack(tensors, library, 1)

    def right_batch_dot(self, X, Y, library=numpy):
        """ performs dense-sparse vector-matrix batched dot product """
        tensors = []
        for i in range(self.dim):
            tensors.append(sum(X[:, p[0]] * Y[:, j] for j, p in self.position_dic.items() if p[1] == i))
        return stack(tensors, library, 1)

    def b(self, M, library=numpy):
        i2r, r2li = get_constraints(self.position_dic)
        S = M.sum(0)
        b = [sum(S[i] for i in r2li[row]) for row in range(self.dim)]
        tensors = [b[i2r[i]] for i in range(self.param)]
        return stack(tensors, library, 0)

    def b2(self, M, library=numpy):
        i2r, r2li = get_constraints(self.position_dic)
        b = [sum(M[:, i] for i in r2li[row]) for row in range(self.dim)]
        tensors = [b[i2r[i]] for i in range(self.param)]
        return stack(tensors, library, 1)
        
class  UpperTriangular(SparseAlgebra):
    def __init__(self, dim):
        positions = [(i, j) for i in range(dim) for j in range(i, dim)]
        super().__init__(positions, check=False)

class  LowerTriangular(SparseAlgebra):
    def __init__(self, dim):
        positions = [(i, j) for i in range(dim) for j in range(i + 1)]
        super().__init__(positions, check=False)

class DirectSum(MatrixAlgebra):
    def __init__(self, *algebras):
        self.algebras = algebras
        self.param = sum(a.param for a in algebras)
        self.dim = sum(a.dim for a in algebras)
        
        self.identity = numpy.concatenate([a.identity for a in algebras], axis=1)
        
        self.positions = []
        offset = 0
        for a in algebras:
            self.positions += map(lambda x: (x[0] + offset, x[1] + offset), a.positions)
            offset += a.dim

    def b(self, M, library=numpy):
        b_ = []
        beg = 0
        end = 0
        for a in self.algebras:
            end += a.param
            b_.append(a.b(M[:, beg:end], library))
            beg = end
        return library.concatenate(b_ ,axis=0)

    def me_layer_T(self, M, indices):
        tensors = []
        beg = 0
        end = 0
        updates = []
        for a in self.algebras:
            end += a.param
            lefts, update = a.me_layer_T(M[:, beg:end], indices)
            tensors.append(lefts)
            updates = updates + update
            beg = end
        concatenated = T.concatenate(tensors, axis=1)
        return concatenated, updates

    def me_layer(self, M, indices):
        tensors = []
        beg = 0
        end = 0
        updates = []
        for a in self.algebras:
            end += a.param
            lefts, update = a.me_layer(M[:, beg:end], indices)
            tensors.append(lefts)
            updates = updates + update
            beg = end
        concatenated = T.concatenate(tensors, axis=2)
        return concatenated, updates

    def right_batch_dot(self, X, Y, library=numpy):
        tensors = []
        beg1 = 0; beg2 = 0
        end1 = 0; end2 = 0
        for a in self.algebras:
            end1 += a.dim; end2 += a.param;
            tensors.append(a.right_batch_dot(X[:, beg1:end1], Y[:, beg2:end2], library))
            beg1 = end1; beg2 = end2
        return library.concatenate(tensors ,axis=1)

class MatrixModel(object):
    """
    Handles the matrices and readout layer together.
    The behavior depends whether there is a readout layer or not.
    """
    def __init__(self, matrix_algebra):
        self.ma = matrix_algebra

    def forward(self, params, cutoff=float("-inf"), cutoff2=float("-inf"),
                    library=numpy, copy=True):
        M_ = forward(params[0], cutoff=cutoff, library=library, copy=copy)
        if len(params) > 1:
            R_ = forward(params[1], cutoff=cutoff2, library=library, copy=copy)
            return [M_, R_]
        else:
            return [R_]

    def normalize(self, params, cutoff=float("-inf"), cutoff2=float("-inf"),
                    library=numpy, copy=True):
        if len(params) > 1:
            ca = CommutativeAlgebra(1)
            R_ = ca.renormalize_forward(params[1], cutoff=cutoff2,
                    library=library, copy=copy)
            M_ = self.ma.renormalize_forward2(params[0], cutoff=cutoff,
                    library=library, copy=copy)
            return [M_, R_]
        else:
            return [self.ma.renormalize_forward(params[0], cutoff=cutoff,
                    library=library, copy=copy)]

    def renormalize(self, params, cutoff=float("-inf"), cutoff2=float("-inf"),
                    library=numpy, copy=True):
        if len(params) > 1:
            ca = CommutativeAlgebra(1)
            R_ = ca.renormalize(params[1], cutoff=cutoff2,
                    library=library, copy=copy)
            M_ = self.ma.renormalize2(params[0], cutoff=cutoff,
                    library=library, copy=copy)
            return [M_, R_]
        else:
             return [self.ma.renormalize(params[0], cutoff=cutoff,
                    library=library, copy=copy)]

def softmax_layer_T(M, indices, readout):
    """
    operates on [time * param * batch] size array
    returns [time * batch] array of probabilities
    """
    batch, time = indices.shape
    logits = readout.dot(M) # V * time * batch
    V, param = readout.shape
    probs = T.nnet.softmax(logits.dimshuffle(1, 2, 0).reshape((-1, V)))
    new_indices = batch*V*T.arange(time)[:, None] + V*T.arange(batch)[None, :] + indices.transpose()
    return probs.reshape((-1,))[new_indices.reshape((-1,))].reshape((time, batch))

def get_kernel(positions):
    pair2i = {pair: index for index, pair in positions.items()}
    kernel = defaultdict(list)
    for i,j in positions.values():
        for j2, k in positions.values():
            if j == j2 and (i,k) in pair2i:
                kernel[pair2i[(i,k)]].append((pair2i[(i,j)], pair2i[(j,k)]))
    return dict(kernel)

def get_constraints(positions):
    rows = sorted(set(p[0] for p in positions.values()))
    
    # index to row dictionary
    i2r = {i: p[0] for i, p in positions.items()}
    
    # row to list of indices dictionary
    r2li = [[i for i, p in positions.items() if p[0] == row] for row in rows]
    return i2r, r2li

def is_associative(positions):
    for i,j in positions:
        for j2,k in positions:
            for k2,l in positions:
                if j == j2 and k == k2 and (i, l) in positions:
                    if ((i,k) in positions) ^ ((k,l) in positions):
                        return False
    return True;

def is_commutative(positions):
    kernel = get_kernel(get_dict(positions))
    for i,pairs in kernel.items():
        for pair in pairs:
            if pair[0] != pair[1]:
                return False
    return True;

def contains_eye(positions):
    n = max(map(max, positions)) + 1 if len(positions) > 0 else 0;
    for i in range(n):
        if (i, i) not in positions:
            return False
    return True

def is_all_different(x):
    return len(set(x)) == len(x)

def check_positions(positions):
    result = True
    dim = get_dim(positions)
    if type(dim) != int or dim <= 0:
        print("ERROR: dimension should be positive integer!", file=sys.stderr)
        return False
    if not is_all_different(positions):
        print("ERROR: the positions should be all different!", file=sys.stderr)
        result = False
    if not contains_eye(positions):
        print("ERROR: algebra does not contain identity!", file=sys.stderr)
        result = False
    if not is_associative(positions):
        print("ERROR: algebra is not associative!", file=sys.stderr)
        result = False
    if is_commutative(positions):
        print("WARNING: algebra is commutative!", file=sys.stderr)
    return result

def get_dim(positions):
    return max(chain(*positions)) + 1

def get_dict(positions):
    return {i: v for i, v in enumerate(positions)}

def get_positions(dim=3, stripes=1, sym=False, file=""):
    symbols = []
    if len(file) == 0:
        for i in range(dim):
            for j in range(dim):
                if abs(i-j) < stripes:
                    if sym or j >= i:
                        symbols.append((i, j))
    else:
        f = sys.stdin if file == "-" else open(file)
        
        for i, l in enumerate(f):
            for j, c in enumerate(l.strip("\n").strip("\r")):
                if len(c.strip()) == 1:
                    symbols.append((i, j))
    return symbols

def main(args):
        symbols = get_positions(args.dim, args.stripes, args.sym, args.input_filename)
        check_positions(symbols)
        print(get_kernel(get_dict(symbols)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Sparse matrix algebra auxiliary script\n" +
                    "Author: Gabor Borbely, contact: borbely@math.bme.hu\n\n" + 
                    "Reads a pattern and writes the corresponding position indices.\n" + 
                    "Or generates pattern using dimension, stripes and symmetry.\n" + 
                    ascii_art,
                formatter_class=CustomFormatter)

    parser.add_argument('-i', '--input', dest='input_filename', type=str,
                    default="",
                    help='input filename or "-" for stdin')
    
    parser.add_argument('-s', '--stripes', dest='stripes', type=int,
                    default=2,
                    help='number of stripes')
    
    parser.add_argument('--sym', default=False, action="store_true",
                    help='symmetric')

    parser.add_argument('-d', '--dim', dest='dim', type=int,
                    default=3,
                    help='dimension')

    main(parser.parse_args())
