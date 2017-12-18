#coding: utf8

import numpy
import theano
import theano.tensor as T
import sys
from os.path import basename
import time
import argparse
from random import shuffle
from auxiliary import *
from matrix_algebra import *
from multiprocessing import Value, Pipe
from threading import Thread as Worker
from extensions import *
from ast import literal_eval

def build_functions(M, ma, args, R=None):
    indices = T.imatrix("i")
    freqs = T.vector("f")
    
    use_readout = R is not None
    # save the numerical values for later
    numeric_values = [M, R] if use_readout else [M]
    
    # symbolic variables
    M_sym = T.matrix("M")
    R_sym = T.matrix("R")
    params = [M_sym, R_sym] if use_readout else [M_sym]

    regularization = M_sym.mean()
    activeM = (M_sym < args.cutoff).mean()
    activeR = (R_sym < args.cutoff2).mean()

    mm = MatrixModel(ma)
    
    normalized_params = mm.normalize(params, args.cutoff, args.cutoff2, T)
    
    M_augmented = T.concatenate((normalized_params[0], ma.identity), axis=0)
    
    if use_readout:
        # time-1 * batch * dim
        lefts, scan_updates1 = ma.right_stochastic_layer(M_augmented, indices[:, :-1])
        # lefts: time * batch * dim
        lefts = T.concatenate([T.ones((1, indices.shape[0], ma.dim)), lefts], axis=0)
        probs, scan_updates2 = theano.scan(lambda x, i, r: (x*r[i, :]).sum(1),
                                           sequences=(lefts, indices.transpose()),
                                           non_sequences=[normalized_params[1]],
                                           strict=True)

        scan_updates = scan_updates1 + scan_updates2
        
        log_probs = T.log2(probs/ma.dim).sum(0)
    else:
        # don't use readout layer
        if args.transpose:
            # lefts: time * param * batch
            lefts, scan_updates = ma.me_layer_T(M_augmented, indices)
        
            # probs: batch
            probs = lefts[-1].sum(0)/ma.dim
        else:
            # lefts: time * batch * param
            lefts, scan_updates = ma.me_layer(M_augmented, indices)
        
            # probs: batch
            probs = lefts[-1].sum(1)/ma.dim

        log_probs = T.log2(probs)
    
    cost = freqs.dot(T.log2(freqs)-log_probs)*args.factor

    renormalized_params = dict(zip(params, mm.renormalize(params, args.cutoff, args.cutoff2, T)))

    optimizer = eval(args.optimizer)(cost+args.alpha*regularization, *params,
                                 minimize=True, eta=args.learning_rate,
                                 constraints=renormalized_params)
    
    optimizer.init(*numeric_values)
    
    updates = optimizer.updates()

    return_values = [cost,
                     activeM if args.cutoff > float("-inf") else T.zeros(()),
                     activeR if args.cutoff2 > float("-inf") else T.zeros(()),
                    ]

    print_flush("3")
    getter_f = optimizer.get_vars()
    
    print_flush("2")
    cost_f = theano.function([indices, freqs], return_values,
                            givens=optimizer.givens(), updates=scan_updates)

    print_flush("1")
    update_f = theano.function(
                    [indices, freqs], return_values,
                    givens=optimizer.givens(),
                    updates=scan_updates + updates)
    
    return cost_f, update_f, getter_f

def print_parameters(getter_f, output_filename, vocab, positions):
    """writes text model file in word2vec-like format"""
    with open(output_filename, "wb") as f:
        f.write(bytes(repr(positions) + "\n", "ascii"))
        M = numpy.concatenate(getter_f(), axis=1)
        for i in range(M.shape[0]):
            f.write(vocab[i] + b" ")
            numpy.savetxt(f, M[i, :].reshape((1,-1)))

def save_model(getter, model_name, vocab, positions):
    print_flush("[Saving model ...", file=sys.stderr)
    print_parameters(getter, model_name, vocab, positions)
    print("Done]", file=sys.stderr)

def read_model(model_name):
    """reads model from text format"""
    with open(model_name, "rb") as f:
        positions = literal_eval(f.readline().strip().decode('ascii'))
        file_pos = f.tell()
        # TODO delimiter could be other than space
        V = numpy.loadtxt(f, dtype=bytes, delimiter=' ', comments=None, usecols=0)
        V = {w: i for i, w in enumerate(V)}
        f.seek(file_pos)
        M = numpy.loadtxt(f, dtype=theano.config.floatX, delimiter=' ',
                            comments=None, converters={0: lambda x: 0.0})
    return V, M[:, 1:], positions

def reader_process(input_queue_send, dataset_reader, splitter, v_reader, args):
    """
    splits the input into batches and sends them to the pipe
    """
    for l in grouper(dataset_reader, args.batch):
        sentences = []
        freqs = []
        max_len = 0
        for line in l:
            parts = line.split(b'\t')
            sentence = splitter(parts[0])
            if len(sentence) >= args.min and len(sentence) <= args.max:
                if len(sentence) > max_len:
                    max_len = len(sentence)
                sentences.append(list(map(v_reader, sentence)))
                if len(parts) > 1:
                    freqs.append(float(parts[1]))
                else:
                    freqs.append(1.0)
        for i in range(len(sentences)):
            sentences[i] += [-1] * (max_len - len(sentences[i]))
        if halted.value > 0:
            break
        if len(sentences) > 0:
            freqs = numpy.array(freqs).astype(theano.config.floatX)
            freq_sum = freqs.sum()
            indices = numpy.array(sentences).astype("int32")
            input_queue_send.send((indices, freqs, freq_sum))
    input_queue_send.send((None,))
    input_queue_send.close()

def worker_process(input_queue_recv, f, getter, args, vocab, positions):
    """
    processes the batches from the pipe
    """
    sample_count = 0
    total_cost = 0
    total_weight = 0.0
    input = input_queue_recv.recv()
    start_time = time.time()
    while len(input) == 3 and halted.value == 0:
        t = time.perf_counter()
        batch_size = len(input[1])
        sample_count += batch_size
        cost, activeM, activeR = f(input[0], input[1])
        total_cost += cost
        total_weight += input[2]
        if not numpy.isfinite(cost):
            halted.value = 1
        print("%d\t%g\t%10d\t%g%%\t%g%%\t" % (sample_count,
                                    total_cost/total_weight,
                                    batch_size/(time.perf_counter()-t),
                                    100*activeM, 100*activeR)
              , end="\r")
        if time.time() - start_time >= args.save_interval:
            print("")
            save_model(getter, args.saved_model, vocab, positions)
            start_time = time.time()
        input = input_queue_recv.recv()
    print("")

def init_model(args):
    if args.initial_model != "":
        print_flush("Loading model from \"" + args.initial_model + "\" ...")
        V, M, positions = read_model(args.initial_model)
        if args.algebra in ["", "load", "Load", "sparse", "Sparse", "SparseAlgebra"]:
            ma = SparseAlgebra(positions)
        else:
            ma = eval(args.algebra)
            if ma.positions != positions:
                print("loaded model positions not consistent with "
                      "the given matrix algebra!")
                print(args.algebra, "!=", positions)
                exit(1)
        if args.use_readout:
            R = M[:, ma.param:]
            M = M[:, :ma.param]
        else:
            R = None
    else:
        print_flush("Initializing random model ...")
        with open(args.vocabfile, "rb") as vocab:
            V = {l.strip().split()[0]: i for i, l in enumerate(vocab)}
        
        ma = eval(args.algebra)
        M = numpy.log2(1.0-numpy.random.rand(len(V), ma.param))
        M = M.astype(theano.config.floatX, copy=False)

        if args.use_readout:
            R = numpy.log2(1.0-numpy.random.rand(len(V), ma.dim))
            R = R.astype(theano.config.floatX, copy=False)
        else:
            R = None

    if args.cutoff.lower() == "none" or args.cutoff is None:
        args.cutoff = float(-2*numpy.log2(M.shape[0]))
    else:
        args.cutoff = float(args.cutoff)

    args.dim = ma.dim
    args.param = ma.param
    args.vocab_size = len(V)
    args.floatX = theano.config.floatX
    
    return M, R, V, ma
    
def main(args):
    M, R, V, ma = init_model(args)
    
    input_basename = basename(args.input_filename)
    
    splitter = do_splitter(literal_eval(args.delimiter))

    unk_index = V[literal_eval(args.unk)]
    v_reader = lambda w: V[w] if w in V else unk_index

    i2w = {v: k for k, v in V.items()}
    
    initial_basename = basename(args.initial_model)
    args.saved_model = args.save_format.format(
            initial_basename, input_basename,
            len(V), args.dim, args.param)
    evaluated_model = initial_basename
    print_flush("Done\n")
    
    for variable in sorted(vars(args)):
        print(variable + ":", vars(args)[variable])

    print_flush("Building functions ...")
    cost_f, update_f, getter_f = build_functions(M, ma, args, R)
    print_flush("Done\n")

    if args.random:
        print_flush("Reading \"" + args.input_filename + "\" ...")
        with open(args.input_filename) as input_f:
            train_database = [l.strip() for l in input_f.buffer]
        print_flush(len(train_database), "samples. Done\n")
    
    print("[sample\tentropy\tsample/sec\tmatrix cutoff\treadout cutoff]")

    global halted
    halted = Value('i', 0)
    
    eval_epoch = False; first_training = True
    for i in range(args.epoch + int(args.do_eval)):
        if i == args.epoch:
            eval_epoch = True
            first_training = False
        elif i == 0:
            first_training = True
            eval_epoch = False
        else:
            first_training = False
            eval_epoch = False
        
        if eval_epoch:
            print("Evaluating: \"" + evaluated_model + "\"")
            f = cost_f
        if first_training:
            print("Training: \"" + args.saved_model + "\"")
            f = update_f
        if not eval_epoch and args.random:
            print_flush("[Shuffle ...", file=sys.stderr)
            shuffle(train_database)
            print("Done]", file=sys.stderr)

        if args.random:
            dataset_reader = iter(train_database)
        else:
            input_f = open(args.input_filename)
            dataset_reader = (l.strip() for l in input_f.buffer)

        input_queue_recv, input_queue_send = Pipe()
        reader = Worker(target=reader_process, name='reader',
                    args=(input_queue_send, dataset_reader, splitter,
                         v_reader, args))
        reader.start()
        print("[Epoch %d]" % (i + 1), file=sys.stderr)
        try:
            worker_process(input_queue_recv, f, getter_f, args, i2w, ma.positions)
        except KeyboardInterrupt:
            halted.value = 1
        reader.join()
        
        if halted.value > 0:
            print("Halted", file=sys.stderr)
            return 1
        if eval_epoch:
            return 0
        if not args.save_at_end or i == args.epoch - 1:
            save_model(getter_f, args.saved_model, i2w, ma.positions)
            evaluated_model = args.saved_model
    return 0

if __name__ == "__main__":
    author = "author: Gábor Borbély"
    contact = "contact: borbely@math.bme.hu"
    parser = argparse.ArgumentParser(
                description="Language Modelling with Matrix Embeddings\n\n" + \
                    ascii_art.replace(" "*len(author), author, 1).replace("\n" +
                    " "*len(contact), "\n" + contact, 1),
                formatter_class=CustomFormatter)

    parser.add_argument('-i', '--input', dest='input_filename', type=str,
                    metavar="FILENAME",
                    help='input filename, format: \"tokens[\\tfreq]\\n\" per line. ' +
                    'Tokens will be split by delimiter. ' + 
                    'Optionally the frequency of the sequence after a tab character.')

    parser.add_argument('-o', '--optimizer', dest='optimizer', type=str,
                    default="AdagradOptimizer", choices=["AdagradOptimizer", "GradientDescentOptimizer"])
                    
    parser.add_argument('-l', '--eta', dest="learning_rate", type=float,
                    default=1.0, metavar="FLOAT",
                    help='initial learning rate of adagrad')

    parser.add_argument('-s', '--save', dest="save_format", type=str,
                    default="{1}.vocab{2}.dim{3}.param{4}.me",
                    metavar="FORMAT",
                    help='format for saving the model file, variables:\n\
                    {0} loaded model name,\n\
                    {1} input file name,\n\
                    {2} size of the vocabulary,\n\
                    {3} dimension of the matrices,\n\
                    {4} number of non-zeros in the matrices')

    parser.add_argument('-v', '--vocab', dest="vocabfile", type=str,
                    default="", metavar="FILENAME",
                    help='vocabulary to learn. Not needed if a previous model is provided to start with.')

    parser.add_argument('--load', dest="initial_model", type=str,
                    default="", metavar="FILENAME",
                    help='loads text model file or initializes a new one if empty')

    parser.add_argument('-e', '--epoch', dest="epoch", type=int,
                    default=1, metavar="INT",
                    help='number of epochs')
    
    parser.add_argument('-m', '--max', dest="max", type=int,
                    default=50, metavar="INT",
                    help='maximum number of tokens in one line')
    
    parser.add_argument('--min', dest="min", type=int,
                    default=1, metavar="INT",
                    help='minimum number of tokens in one line')
                    
    parser.add_argument('-b', '--batch', dest="batch", type=int,
                    default=10000, metavar="INT",
                    help='size of minibatch')
                    
    parser.add_argument('-r', '--random', '--read', dest="random",
                    default=False, action='store_true',
                    help='read the whole database at once and shuffles it between epochs')

    parser.add_argument('--eval', dest="do_eval", action='store_true',
                    default=False,
                    help='after the training, ' + 
                    'an additional epoch is performed just to evaluate, ' + 
                    'only entropy evaluation of the model')

    parser.add_argument('-f', "--factor", dest="factor", type=float,
                    default=1.0, metavar="FLOAT",
                    help='multiplies the objective function by this factor')
    
    parser.add_argument("-c", "--cutoff", dest="cutoff", type=str, default="-inf",
                    metavar="FLOAT",
                    help='in the matrices, 2 to this power is so small ' +
                    'that it is considered 0. It can be None, which means that ' + 
                    'cutoff = -2*log2(vocab_size), which is a reasonable cutoff.')
    
    parser.add_argument("-c2", "--cutoff2", dest="cutoff2", type=float,
                    default="-inf", metavar="FLOAT",
                    help='cutoff of readout, if any')
    
    parser.add_argument('-a', "--algebra", dest="algebra", type=str,
                    default="SparseAlgebra([(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)])",
                    help="sparsity structure, see 'matrix_algebra.py'!")

    parser.add_argument('--save_at_end', dest='save_at_end', action='store_true',
                    default=False,
                    help='saves only at the end of the run')
    
    parser.add_argument('--save_interval', dest='save_interval', type=float,
                    default=float('inf'), metavar="FLOAT",
                    help='Saves regurarly, specified in seconds. ' + 
                    'If not positive then the model is saved after every update.')
                    
    parser.add_argument('-d', '--delimiter', dest='delimiter', type=str,
                    default="b' '", metavar='BYTES',
                    help='delimiter between tokens in sentences, ' + 
                    "it can be empty string for letter-based model")

    parser.add_argument('-u', '--unk', dest='unk', type=str,
                    default="b'<UNK>'", metavar='BYTES',
                    help="special token to mark out-of-vocabulary tokens. "
                    "It should evaluate to a byte array!")

    parser.add_argument('-T', '--transpose', dest='transpose', type=str2bool,
                    default=False, metavar='BOOL',
                    help="transpose the computation.")

    parser.add_argument('-R', '--readout', dest='use_readout', type=str2bool,
                    default=False, metavar='BOOL',
                    help="Use continuous WFSA model "
                        "The resulted model is not compatible to the non-WFSA model!")
                    
    parser.add_argument('--alpha', dest="alpha", type=float, metavar="FLOAT",
                    default=0.0, help='regularization term')

    exit(main(parser.parse_args()))