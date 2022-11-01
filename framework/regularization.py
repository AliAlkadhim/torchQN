import numpy as np
from numba import njit


class L1(object):
    def __init__(self, regularization_constant=int(1e-2) ):
        self.regularization_constant = regularization_constant

    # @njit doesnt applt here
    def update(self, layer):
        weights = layer.weights
        delta = self.regularization_constant * layer.learning_rate
        
        #nudge really small weights to 0
        weights[np.where(np.abs(weights) < delta) ] = 0
        #np.where(condition): when condition is true. For example try the following: x=np.arange(10); x[np.where(x>3)]=0; print(x)

        #nudge positive weights to negative, and negative weights to positive, both by delta
        weights[np.where(weights > 0) ] -= delta
        weights[np.where(weights  < 0) ] += delta
        return weights


class L2(object):
    def __init(self, regularization_constant=int(1e-2) ):
        self.regularization_constant=regularization_constant

    # @njit doesnt apply here
    def update(self, layer):
        weights = layer.weights
        delta  = 2 * self.regularization_constant * layer.learning_rate * weights
        return weights - delta

class WeightLimit(object):
    def __init__(self, weight_maximum=1):
        self.weight_maximum=weight_maximum

    # @njit doesnt apply here
    def update(self, weights):
        weights = layer.weights
        weights[np.where(weights > self.weight_maximum) ] = self.weight_maximum
        weights[np.where(weights <  -self.weight_maximum) ] = - self.weight_maximum
        return weights