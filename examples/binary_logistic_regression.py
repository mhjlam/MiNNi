import snn
import snn.data
import snn.loss
import snn.layer
import snn.model
import snn.accuracy
import snn.optimizer
import snn.activation

X, y = snn.data.generate_spiral(samples=100, classes=2)
X_test, y_test = snn.data.generate_spiral(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = snn.model.Model()
model.add(snn.layer.Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(64, 1))
model.add(snn.activation.Sigmoid())

model.set(loss=snn.loss.BinaryCrossEntropy(), 
          optimizer=snn.optimizer.Adam(decay=5e-7),
          accuracy=snn.accuracy.Categorical(binary=True))

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_freq=100)

'''
epoch: 10000, acc: 0.905, loss: 0.301, (data_loss: 0.264, reg_loss: 0.038), lr: 0.0009950253706593885
validation, acc: 0.790, loss: 0.398
'''
