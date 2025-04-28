import enum
import numpy

NUMPY_32BIT = False


class Metric(enum.Enum):
    REGRESSION = 1
    BINARY = 2
    MULTICLASS = 3


def init():
    # numpy print formatter
    np_fmt = "{: .8f}".format
    numpy.set_printoptions(formatter={'float_kind':np_fmt, 'int_kind':np_fmt})
    
    if NUMPY_32BIT:
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

init.done = False
