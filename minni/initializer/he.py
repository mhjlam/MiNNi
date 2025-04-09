import inspect
from .initializer import Initializer

class He(Initializer):
    def __init__(self):
        pass
    
    def weights(self, F_in, F_out):
        raise Exception(f'TODO: implement function \"{inspect.stack()[0][3]}\"')

    def biases(self, F_in, F_out):
        raise Exception(f'TODO: implement function \"{inspect.stack()[0][3]}\"')
