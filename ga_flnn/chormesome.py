import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt

class Chormesome:
    def __init__(self, n, w = None, activation = 0):
        self.w = np.random.uniform(low = -1, high = 1, size=(n + 1, 1))

        if w is not None:
            self.w = w
        self.activation = activation

    def activation_output(self, z):
        if self.activation == 0:
            a = z
        elif self.activation == 1:
            a = self.tanh(z)
        elif self.activation == 2:
            a = self.relu(z)
        elif self.activation == 3:    
            a = self.elu(z)
        return a

    def compute_fitness(self, X, y):
        w, b = self.w[:-1, :], self.w[[-1], :]
        z = np.dot(X, w) + b
        a = self.activation_output(z)
        mae = mean_absolute_error(y, a)

        self.fitness = 1 / mae
        return self.fitness
    
    def tanh(self, x):
        exp_plus = np.exp(x)
        exp_minus = np.exp(-x)

        return (exp_plus - exp_minus) / (exp_plus + exp_minus)

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def elu(self, x, alpha = 0.5):
        return np.where(x < 0, alpha * (np.exp(x) - 1), x)

    def predict(self, X):
        w, b = self.w[:-1, :], self.w[[-1], :]
        z = np.dot(X, w) + b
        a = self.activation_output(z)
        return a
