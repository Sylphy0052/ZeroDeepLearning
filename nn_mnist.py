import numpy as np
import os
import pickle

from function import sigmoid, softmax
from mnist import load_mnist


class NNMnist:
    def __init__(self):
        self._get_data()
        self._init_network()

    def _get_data(self):
        self.x_train, self.y_train, self.x_test, self.y_test = \
            load_mnist(normalize=True, flatten=True, one_hot_label=False)

    def _init_network(self):
        model_path = os.path.join('models', 'sample_weight.pkl')
        with open(model_path, 'rb') as f:
            self.network = pickle.load(f)

    def get_train_data(self):
        return self.x_train, self.y_train

    def predict(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, w3) + b3
        y = softmax(a3)

        return y
