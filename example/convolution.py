import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import minni
import minni.activator
import minni.initializer
import minni.layer
import minni.loss
import minni.model
import minni.optimizer

minni.init()

if __name__ == '__main__':
    model = minni.model.Model(loss=minni.loss.CategoricalCrossEntropy(), optimizer=minni.optimizer.Adam())
    model.add(minni.layer.Conv(3, 16, kernel_size=3, stride=1, padding=1, initializer=minni.initializer.Random()))
    model.add(minni.activator.Rectifier())
    model.add(minni.layer.MaxPooling(pool_size=2, stride=2))
    model.add(minni.layer.Flatten())
    model.add(minni.layer.Dense(16*16*16, 10, activator=minni.Softmax()))
