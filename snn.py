import numpy

class Gradient:
    def __init__(self, dweights, dbiases, dinputs):
        self.dweights = dweights
        self.dbiases = dbiases
        self.dinputs = dinputs

class Regularizer:
    def __init__(self, l1w=0, l1b=0, l2w=0, l2b=0):
        self.l1w = l1w
        self.l1b = l1b
        self.l2w = l2w
        self.l2b = l2b

def init():
    if not init.done:
        # numpy print formatter
        np_fmt = "{: .8f}".format
        numpy.set_printoptions(formatter={'float_kind':np_fmt, 'int_kind':np_fmt})
        
        # numpy dot
        np_dot = numpy.dot
        def dot(*args, **kwargs):
            return np_dot(*[a.astype('float64') for a in args], **kwargs).astype('float32')
        numpy.dot = dot

        # numpy zeros
        np_zeros = numpy.zeros
        def zeros(*args, **kwargs):
            if len(args) <= 1 and 'dtype' not in kwargs:
                kwargs['dtype'] = 'float32'
            return np_zeros(*args, **kwargs)
        numpy.zeros = zeros

        # numpy randn
        np_randn = numpy.random.randn
        def randn(*args, **kwargs):
            return np_randn(*args, **kwargs).astype('float32')
        numpy.random.randn = randn

        # numpy eye
        np_eye = numpy.eye
        def eye(*args, **kwargs):
            if len(args) <= 1 and 'dtype' not in kwargs:
                kwargs['dtype'] = 'float32'
            return np_eye(*args, **kwargs)
        numpy.eye = eye

        # numpy seed
        numpy.random.seed(0)
         
        init.done = True
        print('SNN initialized')

init.done = False
init()
