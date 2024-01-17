import numpy

class Layer:
   def __init__(self, neurons):
      self.neurons = neurons
   
   def __str__(self, n=0):
      label = f'Layer {str(n)}:\n' if n >= 0 else 'Layer:\n'
      return label + '  type: Generic\n\n' + \
                     '\n'.join(['  ' + '  '.join(n.str(i).splitlines(True)) 
                                for i, n in enumerate(self.neurons)])

   def str(self, n=0):
      try:
         return self.__str__(n)
      except:
         return self.__str__()

   def forward(self, inputs):
      return numpy.dot(inputs, self.weights) + self.biases
