#! /usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import configparser
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from PIL import Image

from function import *
from perceptron import Perceptron
from mnist import load_mnist
from nn_mnist import NNMnist
from gradient import GradientNet
from two_layer_net import TwoLayerNet
from util import __train, im2col


config = configparser.ConfigParser()
config.read('setting.ini', encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('type', type=str)


def print_output(model, opt):
    print('{}(0, 0): {}'.format(opt, model.predict(np.array([0, 0]))))
    print('{}(0, 1): {}'.format(opt, model.predict(np.array([0, 1]))))
    print('{}(1, 0): {}'.format(opt, model.predict(np.array([1, 0]))))
    print('{}(1, 1): {}'.format(opt, model.predict(np.array([1, 1]))))


def perceptron1(model):
    # AND
    model.fit(w=np.array([0.5, 0.5]), b=-0.7)
    print_output(model, 'AND')

    # OR
    model.fit(w=np.array([0.5, 0.5]), b=-0.2)
    print_output(model, 'OR')

    # NAND
    model.fit(w=np.array([-0.5, -0.5]), b=0.7)
    print_output(model, 'NAND')


def plot_func(func):
    title = func.__name__
    x = np.arange(-5.0, 5.0, 0.1)
    y = func(x)
    plt.plot(x, y)
    plt.title(title)
    plt.show()


def func_test():
    # print(sigmoid(np.array([-1, 1, 2])))
    # print(step_function(np.array([-1, 1, 2])))
    # print(relu(np.array([-1, 1, 2])))

    # plot_func(sigmoid)
    # plot_func(step_function)
    # plot_func(relu)
    plot_func(softmax)


def show_img(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def mnist():
    x_train, y_train, x_test, y_test = \
        load_mnist(flatten=True, normalize=False)

    # print(f'{x_train.shape=}')
    # print(f'{y_train.shape=}')
    # print(f'{x_test.shape=}')
    # print(f'{y_test.shape=}')

    img = x_train[0]
    label = y_train[0]
    print(f'{label=}')
    print(f'{img.shape=}')
    img = img.reshape(28, 28)
    show_img(img)


def mnist_nn():
    nm = NNMnist()
    x, l = nm.get_train_data()
    accuracy_cnt = 0
    batch_size = 100
    for i in tqdm(range(0, len(x), batch_size)):
        x_batch = x[i: i+batch_size]
        y_batch = nm.predict(x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == l[i:i+batch_size])

    print('Accuracy: {}'.format(str(float(accuracy_cnt) / len(x))))


def mnist_train():
    # x_train, y_train, x_test, y_test = \
    #     load_mnist(flatten=True, normalize=True, one_hot_label=True)

    # print(f'{x_train.shape=}')
    # print(f'{y_train.shape=}')

    # train_size = x_train.shape[0]
    # batch_size = 10
    # batch_mask = np.random.choice(train_size, batch_size)
    # x_batch = x_train[batch_mask]
    # y_batch = y_train[batch_mask]

    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    def f(x): return np.sum(x * x)
    grad = numerical_gradient(f, np.array([X, Y]).T).T

    plt.figure()
    plt.quiver(X, Y, -grdad[0], -grad[1], angles='xy')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()


def gradient_net():
    nn = GradientNet()
    print(nn.w)
    x = np.array([0.6, 0.9])
    p = nn.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0, 0, 1])
    print(nn.loss(x, t))

    dw = numerical_gradient(lambda w: nn.loss(x, t), nn.w)
    print(dw)


def two_layer_net():
    x_train, y_train, x_test, y_test = \
        load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    iter_per_epoch = max(train_size / batch_size, 1)

    # 784 = 28 x 28 画像をflattenした
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]

        grad = net.gradient(x_batch, y_batch)
        for key in ['W1', 'b1', 'W2', 'b2']:
            net.params[key] -= learning_rate * grad[key]

        loss = net.loss(x_batch, y_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, y_train)
            test_acc = net.accuracy(x_test, y_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f'{train_acc=}\n{test_acc=}')


def gradient_check():
    x_train, y_train, x_test, y_test = \
        load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = x_train[:3]
    y_batch = y_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, y_batch)
    grad_backprop = network.gradient(x_batch, y_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f'{key} : {diff}')


def weight_init_activation():
    x = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    activations = {}

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]

        # w = np.random.randn(node_num, node_num) * 1
        # w = np.random.randn(node_num, node_num) * 0.01
        # Xavierの初期値
        w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

        z = np.dot(x, w)
        a = sigmoid(z)
        activations[i] = a

    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1) + '-layer')
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()


def batch_norm():
    x_train, y_train, x_test, y_test = \
        load_mnist(normalize=True)

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    max_epochs = 20
    # train_size = x_train.shape[0]
    # batch_size = 100
    # learning_rate = 0.01

    weight_scale_list = np.logspace(0, -4, num=16)
    x = np.arange(max_epochs)

    for i, w in enumerate(weight_scale_list):
        print(f'---{i+1} / 16---')
        train_acc_list, bn_train_acc_list = __train(x_train, y_train, w)
        plt.subplot(4, 4, i+1)
        plt.title(f'W: {w}')
        if i == 15:
            plt.plot(x, bn_train_acc_list,
                     label='Batch Normalization', markevery=2)
            plt.plot(x, train_acc_list, linestyle='--',
                     label='Normal(without BatchNorm)', markevery=2)
        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, linestyle='--', markevery=2)

        plt.ylim(0, 1.0)
        if i % 4:
            plt.yticks([])
        else:
            plt.ylabel('accuracy')
        if i < 12:
            plt.xticks([])
        else:
            plt.xlabel('epochs')
        plt.legend(loc='lower right')

    os.makedirs('figs', exist_ok=True)
    plt.savefig('./figs/batch_norm.png')


def cnn():
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape)


def main(args):
    if args.type in ['perceptron', 'p']:
        model = Perceptron()
        perceptron1(model)
    elif args.type in ['function', 'f']:
        func_test()
    elif args.type in ['mnist', 'm']:
        mnist()
    elif args.type in ['mnist_nn', 'mn']:
        mnist_nn()
    elif args.type in ['nn_train', 'nt']:
        mnist_train()
    elif args.type in ['gradient_nn', 'gn']:
        gradient_net()
    elif args.type in ['two_layer_net', 'tln']:
        two_layer_net()
    elif args.type in ['gradient_check', 'gc']:
        gradient_check()
    elif args.type in ['weight_init_activation', 'wia']:
        weight_init_activation()
    elif args.type in ['batch_norm', 'bn']:
        batch_norm()
    elif args.type in ['cnn', 'c']:
        cnn()
    else:
        print('Put some types...(ex. perceptron)')
        sys.exit(1)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
