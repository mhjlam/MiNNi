import snn
import snn.data
import snn.loss
import snn.layer
import snn.model
import snn.accuracy
import snn.optimizer
import snn.activation

X, y = snn.data.generate_sine(samples=1000)

model = snn.model.Model()
model.add(snn.layer.Dense(1, 64, weight_scale=0.1))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(64, 64, weight_scale=0.1))
model.add(snn.activation.ReLU())
model.add(snn.layer.Dense(64, 1, weight_scale=0.1))
model.add(snn.activation.Linear())

model.set(loss_func=snn.loss.MeanSquaredError(), 
          optimizer=snn.optimizer.Adam(learning_rate=0.005, decay=1e-3),
          accuracy=snn.accuracy.RegressionAccuracy())

model.finalize()

model.train(X, y, epochs=10000, print_freq=100)

'''
epoch: 10000, acc: 0.905, loss: 0.301, (data_loss: 0.264, reg_loss: 0.038), lr: 0.0009950253706593885
validation, acc: 0.790, loss: 0.398
'''
