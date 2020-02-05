import numpy as np
from collections import OrderedDict

from function import (cross_entropy_error,
                      numerical_gradient,
                      sigmoid,
                      softmax,
                      sigmoid_grad)
from layer import (Affine,
                   Relu,
                   SoftmaxWithLoss)


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
        # w1, w2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']

        # a1 = np.dot(x, w1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, w2) + b2
        # y = softmax(a2)

        # return y

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
        # return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_w(w): return self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dw, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dw, self.layers['Affine2'].db

        return grads

        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
        # grads = {}

        # batch_num = x.shape[0]

        # # forward
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)

        # # backward
        # dy = (y - t) / batch_num
        # grads['W2'] = np.dot(z1.T, dy)
        # grads['b2'] = np.sum(dy, axis=0)

        # dz1 = np.dot(dy, W2.T)
        # da1 = sigmoid_grad(a1) * dz1
        # grads['W1'] = np.dot(x.T, da1)
        # grads['b1'] = np.sum(da1, axis=0)

        # return grads
