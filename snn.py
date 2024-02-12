import numpy
from collections import namedtuple

Gradient = namedtuple('Gradient', 'dweights dbiases dinputs')
Regularizer = namedtuple('Regularizer', 'l1w l1b l2w l2b')

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

# Calculate accuracy from softmax output and targets
def accuracy(predictions, targets):
    predictions = numpy.argmax(predictions, axis=1)
    if len(targets.shape) == 2:
        targets = numpy.argmax(targets, axis=1)
    return numpy.mean(predictions == targets)

def show_epoch_stats(epoch, accuracy, loss, data_loss, regularization_loss, learning_rate):
    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' +
              f'reg_loss: {regularization_loss:.3f}), ' +
              f'lr: {learning_rate}')

def init():
    if not init.done:
        numpy.random.seed(0)
        setattr(numpy, 'zeros', numpy_zeros)
        setattr(numpy.random, 'randn', numpy_randn)
        setattr(numpy, 'eye', numpy_eye)
        
        np_fmt = "{: .8f}".format
        numpy.set_printoptions(formatter={'float_kind':np_fmt, 'int_kind':np_fmt})
        
        init.done = True
        print('SNN initialized')

init.done = False
init()
