import numpy as np


class Perceptron:
    def __init__(self):
        pass

    def fit(self, w=np.array([]), b=1.0):
        self.w = w
        self.b = b

    def predict(self, X):
        return 1 if np.sum(self.w * X) + self.b > 0 else 0
