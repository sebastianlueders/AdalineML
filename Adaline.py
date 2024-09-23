#!/usr/bin/env python3

import numpy as np

class Adaline:
    
    def __init__(self, lr=0.01, epochs=50, rand_seed=1):
        
        self.lr = lr
        self.epochs = epochs
        self.rand_seed = rand_seed


    def fit(self, t_data, targets):

        rand_gen = np.random.RandomState(self.rand_seed)  
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + t_data.shape[1])

        self.costs_ = []

        for i in range(self.epochs):
            net_input = self.net_input(t_data)
            output = self.activation(net_input)
            errors = (targets - output)
            self.weights_[1:] += self.lr * t_data.T.dot(errors)
            self.weights_[0] += self.lr * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.costs_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
