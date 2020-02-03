#! /usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import configparser
import numpy as np
import sys

from .function import *
from .perceptron import Perceptron


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


def func_test():
    print(sigmoid(1))


def main(args):
    if args.type in ['perceptron', 'p']:
        model = Perceptron()
        perceptron1(model)
    elif args.type in ['function', 'f']:
        func_test()
    else:
        print('Put some types...(ex. perceptron)')
        sys.exit(1)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
