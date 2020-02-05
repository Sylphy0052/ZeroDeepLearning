import numpy as np
from function import softmax, cross_entropy_error


class GradientNet:
    def __init__(self):
        self.w = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.w)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
