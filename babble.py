# -*- coding: utf8 -*-

import sys
from os import environ
if 'THEANO_FLAGS' in environ:
    environ['THEANO_FLAGS'] += ',device=cpu'
else:
    environ['THEANO_FLAGS'] = 'device=cpu'

import argparse
import numpy
from lm_me import read_model
from auxiliary import *
from matrix_algebra import *
from ast import literal_eval

def predict_words(left, R, random):
    probs = left.dot(R)
    indices=numpy.zeros(probs.shape[0], dtype=numpy.int32)
    if random:
        indices = numpy.array([numpy.argmax(numpy.random.multinomial(1, probs[i])) for i in range(len(probs))])
    else:
        numpy.argmax(probs, axis=1, out=indices)
    predictedprobs = probs.reshape((-1,))[indices + numpy.arange(probs.shape[0])*probs.shape[1]]
    return indices, -numpy.log10(predictedprobs)

def end_sentences(indices, M, R, stop_index, max, ma, vocab, random):
    indices = numpy.array(indices, dtype=numpy.int32)
    batch_size = indices.shape[0]
    stop = stop_index*numpy.ones((batch_size, 1), dtype=numpy.int32)
    blank = -numpy.ones((batch_size, 1), dtype=numpy.int32)
    
    probs = [[] for _ in range(batch_size)]
    
    left_ = numpy.ones((batch_size, ma.dim))/ma.dim
    if indices.shape[1] > 0:
        for i in indices.transpose():
            i_ = ((i != -1)).nonzero()[0]
            probs_i = left_[i_].dot(R).reshape((-1,))[i[i_] + numpy.arange(len(i_))*R.shape[1]]
            for x, j in enumerate(i_):
                probs[j].append(-numpy.log10(probs_i[x]))
            i_ = ((i != stop_index) & (i != -1)).nonzero()[0]
            left_[i_] = ma.right_batch_dot(left_[i_], M[i[i_], :])
        i_ = ((i != stop_index)).nonzero()[0]
    else:
        i_ = numpy.arange(batch_size, dtype=numpy.int32)
    while indices.shape[1] <= max and len(i_) > 0:
        indices = numpy.append(indices, stop, axis=1)
        next_words = indices[:, -1]
        next_words[i_], probs_i = predict_words(left_[i_], R, random)
        for x, j in enumerate(i_):
            probs[j].append(probs_i[x])
        left_[i_] = ma.right_batch_dot(left_[i_], M[next_words[i_], :])
        i_ = (next_words != stop_index).nonzero()[0]
    return indices, probs

def main(args):
    args.unk, args.eos, args.space = map(literal_eval, [args.unk, args.eos, args.space])
    
    print_flush("Reading model ...", file=sys.stderr)
    V, M, positions = read_model(args.model_filename)
    if args.algebra in ["", "load", "Load", "sparse", "Sparse", "SparseAlgebra"]:
        ma = SparseAlgebra(positions)
    else:
        ma = eval(args.algebra)
        if ma.positions != positions:
            print("loaded model positions not consistent with "
                  "the given matrix algebra!")
            print(args.algebra, "!=", positions)
            return 1
    R = M[:, ma.param:]
    M = M[:, :ma.param]

    mm = MatrixModel(ma)
    
    if args.normalize:
        mm.normalize([M, R], cutoff=args.cutoff, cutoff2=args.cutoff2,
                    library=numpy, copy=False)
    else:
        mm.forward([M, R], cutoff=args.cutoff, cutoff2=args.cutoff2,
                    library=numpy, copy=False)

    R = R.transpose()
    
    to_word_vocab = {v: k for k, v in V.items()}
    print("Done", file=sys.stderr)
    
    if args.unk not in V or args.eos not in V:
        print("You have to supply 'unknown' and 'end-of-sentence' words!", file=sys.stderr)
        return 1

    unk_index = V[args.unk]
    eos_index = V[args.eos]
    w2i = lambda w: V[w] if w in V else unk_index
    
    for lines in grouper(sys.stdin.buffer, args.batch_size):
        words = [x.strip().split() for x in lines]
        max_length = max(map(len, words))
        indices = [list(map(w2i,x)) + [-1]*(max_length - len(x)) for x in words]
        indices, probs = end_sentences(indices, M, R, eos_index, args.max, ma,
                                        to_word_vocab, args.random)
        for line, lineprobs in zip(indices, probs):
            for i in line:
                if i != -1:
                    sys.stdout.buffer.write(to_word_vocab[i])
                    if i == eos_index:
                        break
                    else:
                        sys.stdout.buffer.write(args.space)
            if args.log == 1:
                print("\t", file=sys.stderr, end="")
                print(sum(lineprobs), file=sys.stderr)
            elif args.log == 2:
                print("\t", file=sys.stderr, end="")
                for p in lineprobs:
                    print(p, "", file=sys.stderr, end="")
                print("", file=sys.stderr)
            print("")
    return 0

if __name__ == "__main__":
    author = "author: Gábor Borbély"
    contact = "contact: borbely@math.bme.hu"
    parser = argparse.ArgumentParser(
                description=author + ", " + contact,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-m', '--model', dest='model_filename', type=str,
                    default="",
                    help='the embedding file')
    parser.add_argument('-u', '--unk', dest='unk', type=str,
                    default="b'unk'", metavar='bytes',
                    help='special token to mark out-of-vocabulary tokens. ' + 
                    "It should evaluate to a byte array!")
    parser.add_argument('--max', dest="max", type=int,
                    default=50,
                    help='maximum number of tokens in the output')
    parser.add_argument('-b', '--batch', dest="batch_size", type=int,
                    default=10,
                    help='batch size')

    parser.add_argument('-a', "--algebra", dest="algebra", type=str,
                    default="load",
                    help="sparsity structure, unless you state otherwise " + 
                        "an inefficient sparse model is used!")

    parser.add_argument('-n', '--normalize', dest='normalize', action='store_true',
                        default=False,
                        help="Normalizes the model to a probabilistic one")
    parser.add_argument('-eos', '--eos', dest='eos', type=str,
                    default="b'eos'", metavar='bytes',
                    help='special token to mark end-of-sentence. ' + 
                    "It should evaluate to a byte array!")
    parser.add_argument('-s', '--space', "-d", "--delimiter",
                    dest='space', type=str, default="b' '", metavar='bytes',
                    help='delimiter between tokens')    
    parser.add_argument("-r", '--random', dest='random', action='store_true',
                        default=False,
                        help="Performs random sampling according to the calculated probabilities." + 
                        " If False, then the most probable outcome will be returned.")
    parser.add_argument("-l", "--log", dest='log', default=0,
                        choices=[0,1,2], type=int,
                        help="0: don't print to stderr, " + 
                        "1: prints the -log10 probability of the outcome after each line to stderr, " + 
                        "2: prints the -log10 probability of each prediction to stderr")
    parser.add_argument("-c", '--check', dest='check', action='store_true',
                        default=False,
                        help="Performs sanity check on the loaded model. " + 
                             "Not needed if you trained your model with lm_me.py!")

    parser.add_argument('--cutoff', dest='cutoff', type=float, default=float('-inf'), 
                        help="cutoff exponent for small matrix values")
    parser.add_argument('--cutoff2', dest='cutoff2', type=float, default=float('-inf'), 
                        help="cutoff exponent for small readout values")
                        
    exit(main(parser.parse_args()))
