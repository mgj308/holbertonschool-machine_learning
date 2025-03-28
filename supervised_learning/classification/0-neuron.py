#!/usr/bin/env python3
'''Module that has class Neuron'''
import numpy as np


class Neuron:
    '''Class Neuron that defines a single neuron performing binary class'''
    def __init__(self, nx):
        '''Class constructor'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
