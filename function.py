import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    return (x > 0).astype(np.int)


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# def cross_entropy_error(y, t):
#     delta = 1e-7
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y + delta)) / batch_size


def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # one-hot-vectorの場合、ラベルに変換する
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def numerical_diff(f, x):
    """[summary]
    数値微分を行う - 微小な差分によって微分を求める

    Arguments:
        f {[type]} -- 微分を行う関数
        x {[type]} -- 変数

    Returns:
        [type] -- 微分した結果
    """
    h = 1e-4
    return (f(x+h) - f(x - h)) / (2 * h)


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


# def numerical_gradient(f, X):
#     if X.ndim == 1:
#         return _numerical_gradient_no_batch(f, X)

#     grad = np.zeros_like(X)
#     for idx, x in enumerate(X):
#         grad[idx] = _numerical_gradient_no_batch(f, x)

#     return grad

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad


def gradient_decent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
