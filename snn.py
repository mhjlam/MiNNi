import numpy
from collections import namedtuple

Gradient = namedtuple('Gradient', 'dweights dbiases dinputs')

npz = getattr(numpy, 'zeros')
npr = getattr(numpy.random, 'randn')
npe = getattr(numpy, 'eye')

def numpy_zeros(*args, **kwargs):
    if len(args) < 2 and 'dtype' not in kwargs:
        kwargs['dtype'] = 'float32'
    return npz(*args, **kwargs)

def numpy_randn(*args, **kwargs):
    return npr(*args, **kwargs).astype('float32')

def numpy_eye(*args, **kwargs):
    if len(args) < 2 and 'dtype' not in kwargs:
        kwargs['dtype'] = 'float32'
    return npe(*args, **kwargs)

def init():
    if not init.done:
        init.done = True
        
        numpy.random.seed(0)
        
        np_fmt = "{: .8f}".format
        numpy.set_printoptions(formatter={'float_kind':np_fmt, 'int_kind':np_fmt})
        
        setattr(numpy, 'zeros', numpy_zeros)
        setattr(numpy.random, 'randn', numpy_randn)
        setattr(numpy, 'eye', numpy_eye)
        print('SNN initialized')

init.done = False
init()
