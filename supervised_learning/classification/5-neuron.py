#!/usr/bin/env python3
'''Module that has class Neuron'''
import numpy as np


class Neuron:
    '''
    Class Neuron that defines a single
    neuron performing binary classification
    '''
    def __init__(self, nx):
        '''Class constructor'''
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Getter function of W'''
        return self.__W

    @property
    def b(self):
        '''Getter function of b'''
        return self.__b

    @property
    def A(self):
        '''Getter function of A'''
        return self.__A

    def forward_prop(self, X):
        '''
        Public method that calculates the forward
        propagation of the neuron
        '''
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        '''
        Calculates cost of the model using logistic
        regresion
        '''
        m = Y.shape[1]
        C = - (1 / m) * np.sum(Y * np.log(A) +
                               (1 - Y) * (np.log(1.0000001 - A)))
        return C

    def evaluate(self, X, Y):
        '''
        Evaluates the neuron’s predictions
        '''
        self.forward_prop(X)
        P = np.where(self.__A >= 0.5, 1, 0)
        return P, self.cost(Y, self.__A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        '''
        Calculates one pass of gradient descent
        '''
        m = Y.shape[1]
        dw = np.matmul(A - Y, X.T) / m
        db = np.sum(A - Y) / m
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
