from itertools import islice, chain
import argparse
import sys
import numpy

ascii_art = (
"                                      [1]\n"
"                                      [1]\n"
"                                      [1]\n"
"        [       ] [ X X X ] [       ]    \n"
"        [  the  ] [   X X ] [ barks ]    \n"
"        [       ] [     X ] [       ]    \n"
"[1 1 1]                                  ")

def grouper(iterable, n, step=1):
    "Collect data into chunks or blocks at size at most n"
    # grouper('ABCDEFG', 3) --> [ABC, DEF, G]
    # step is irrelevant
    while True:
        l = list(islice(iterable, n))
        if len(l) > 0:
            yield l
        else:
            break

def do_splitter(delimiter):
    if len(delimiter) == 0:
        return (lambda x: list(x))
    else:
        return (lambda x: x.split(delimiter))

# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def windower(iterable, n, step=1):
    # Collect data into equal blocks at size n
    # windower('ABCDEFG', 3) --> [ABC, BCD, CDE, DEF, EFG]
    l = list(islice(iterable, n))
    yield l.copy()
    i = 0
    for w in iterable: # what's left of it
        i += 1
        l.append(w)
        del l[0]
        if i % step == 0:
            yield l.copy()
    if i % step != 0:
        yield l.copy()

def word_reader(f, w_delim=b' \t\v', s_delim=b'\n\r\f'):
    c = f.read(1)
    whitespaces = s_delim + w_delim
    while len(c) > 0 and c in whitespaces:
        c = f.read(1)
    while len(c) > 0 and c not in s_delim:
        w = c
        c = f.read(1)
        while c not in whitespaces and len(c) > 0:
            w += c
            c = f.read(1)
        yield w
        while c in w_delim and len(c) > 0:
            c = f.read(1)

def window_reader(f, w, w_delim=b' \t\v', s_delim=b'\n\r\f',
                    sos=b"<S>", eos=b"</S>",
                    use_sliding=False, step=1):
    preamble = [sos] if sos is not None else []
    appendix = [eos] if eos is not None else []
    min_length = (0 if sos is None else 1) + (0 if eos is None else 1)
    sub_iterator = windower if use_sliding else grouper
    if type(w) == int and w > 0:
        while True:
            line = chain(preamble, word_reader(f, w_delim, s_delim), appendix)
            length = 0
            for window in sub_iterator(line, w, step):
                length += len(window)
                yield window
            if length <= min_length:
                break

def print_flush(*args, file=sys.stdout):
    print(*args, file=file, end=" ", flush=True)

def forward(M, cutoff=float("-inf"), library=numpy, copy=True):
    mask = (M >= cutoff)
    if copy == False:
        library.exp2(M, out=M)
        M *= mask
    else:
        return mask * library.exp2(M)
        
class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                        argparse.ArgumentDefaultsHelpFormatter):
    pass
