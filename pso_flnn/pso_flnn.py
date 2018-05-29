import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt

class ExpandData:
    def __init__(self, data, train_idx, test_idx, sliding, expand_func = 1):
        self.data = data
        self.scaler = MinMaxScaler()
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.expand_func = expand_func

    def expand_data(self):
        if self.expand_func == 0:
            return self.chebyshev()
        elif self.expand_func == 1:
            return self.powerseries()
        elif self.expand_func == 2:
            return self.laguerre()
        elif self.expand_func == 3:
            return self.legendre()

    def legendre(self):
        data = self.data

        x1 = data
        x2 = 3 / 2 * np.power(data, 2) - 1 / 2
        x3 = 1 / 3 * (5*data*x2 - 2*x1)
        x4 = 1 / 4 * (7*data*x3 - 3*x2)
        x5 = 1 / 5 * (9*data*x4 - 4*x3)
        x6 = 1 / 6 * (11*data*x5 - 5*x4)

        return [x2, x3, x4, x5, x6] 

    def laguerre(self):
        data = self.data

        x1 = -data + 1
        x2 = np.power(data, 2) / 2 - 2 * data + 1
        x3 = 1 / 3 * ((5 - data)*x2 - 2*x1)
        x4 = 1 / 4 * ((7 - data)*x3 - 3*x2)
        x5 = 1 / 5 * ((9 - data)*x4 - 4*x3)
        x6 = 1 / 6 * ((11 - data)*x5 - 5*x4)

        return [x2, x3, x4, x5, x6]
    
    def chebyshev(self):
        data = self.data

        x1 = data
        x2 = 2 * np.power(data, 2) - 1
        x3 = 4 * np.power(data, 3) - 3 * data
        x4 = 8 * np.power(data, 4) - 8 * np.power(data, 2) + 1
        x5 = 2 * data * x4 - x3
        x6 = 2 * data * x5 - x4

        return [x2, x3, x4, x5, x6]
    
    def powerseries(self):
        data = self.data

        x1 = data
        x2 = np.power(data, 2)
        x3 = np.power(data, 3)
        x4 = np.power(data, 4)
        x5 = np.power(data, 5)
        x6 = np.power(data, 6)

        return [x2, x3, x4, x5, x6]

    def scale(self, expanded_data):
        scale_expanded_data = []

        for ed in expanded_data:
            scale_expanded_data.append(self.scaler.fit_transform(ed))

        return scale_expanded_data

    def process_data(self):
        train_idx, test_idx, sliding = self.train_idx, self.test_idx, self.sliding

        expanded_data = self.expand_data()
        scale_data = self.scale(expanded_data)

        data_expanded = np.ones(self.data[:train_idx + test_idx, :].shape)

        for sd in scale_data:
            for i in range(0, sliding):
                data_expanded = np.concatenate((data_expanded, sd[i:i + train_idx + test_idx, :]), axis=1)

        return data_expanded[:, 1:]

class Particle:
    def __init__(self, n, activation = 0):
        self.n = n
        self.x = np.random.uniform(low = -5, high = 5, size=(n, 1))
        self.v =  np.zeros((n, 1))
        self.pbest = np.copy(self.x)
        self.best_fitness = -1
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
        z = np.dot(X, self.x)

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
        z = np.dot(X, self.x)
        a = self.activation_output(z)
        return a


class Population:
    def __init__(self, s, c1, c2, activation = 0):
        self.s = s
        self.c1 = c1
        self.c2 = c2
        self.c = c1 + c2
        # self.theta = 2 / (self.c - 2 + np.sqrt(np.square(self.c) - 4 * self.c))
        self.activation = activation
        self.Vmax = 10
        self.w = 0.729

    def init_pop(self, d):
        self.population = []
        for i in range(0, self.s):
            c = Particle(d, activation = self.activation)
            self.population.append(c)

    def train(self, X, y, epochs=1000):
        best_particle = None
        best_fitness = -1

        d = X.shape[1]
        self.init_pop(d)

        for e in range(epochs):
            for p in self.population:
                fitness = p.compute_fitness(X, y)

                if fitness > p.best_fitness:
                    p.best_fitness = fitness
                    p.pbest = copy.deepcopy(p.x)
                
                if p.best_fitness > best_fitness:
                    best_fitness = fitness
                    best_particle = copy.deepcopy(p)

            
            print("Epoch {0}: {1}".format(e, best_fitness))

            for p in self.population:
                # v_new = self.theta * (p.v + self.c1 * np.random.uniform(low=0,high=1,size=(p.n, 1)) * (p.pbest - p.x) + self.c2 *  np.random.uniform(low=0,high=1,size=(p.n, 1)) * (best_particle.pbest - p.x))
                v_new = self.w * p.v + self.c1 * np.random.uniform(low=0,high=1,size=(p.n, 1)) * (p.pbest - p.x) + self.c2 *  np.random.uniform(low=0,high=1,size=(p.n, 1)) * (best_particle.pbest - p.x)
                
                v_new[v_new > self.Vmax] = self.Vmax
                v_new[v_new < -self.Vmax] = -self.Vmax
                
                x_new = p.x + v_new

                p.v = copy.deepcopy(v_new)
                p.x = copy.deepcopy(x_new)

        return best_particle

dt = pd.read_csv('data_resource_usage_10Minutes_6176858948.csv', usecols=[3], names=['cpu'])

sliding = 2
train_idx = 3280
test_idx = 820

class Model:
    def __init__(self, data_original, train_idx, test_idx, sliding, expand_func = 0, pop_size = 200, c1 = 2, c2 = 2, activation = 2):
        self.data_original = data_original
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.data = data_original[:train_idx + test_idx + sliding + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.activation = activation

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.real_inverse[:, 0], color='#009FFD', linewidth=2.5)
        plt.plot(self.pred_inverse[:, 0], color='#FFA400', linewidth=2.5)
        plt.ylabel('CPU')
        plt.xlabel('Timestamp')
        plt.legend(['Actual', 'Prediction'], loc='upper right')
        # plt.savefig(self.pathsave + self.filenamesave + ".png")
        plt.show()
        plt.close()

    
    def preprocessing_data(self):
        data, train_idx, test_idx, sliding, expand_func = self.data, self.train_idx, self.test_idx, self.sliding, self.expand_func

        data_scale = self.scaler.fit_transform(data)
        data_transform = data_scale[:train_idx + test_idx, :]

        for i in range(1, sliding+1):
            data_transform = np.concatenate((data_transform, data_scale[i:i+train_idx + test_idx, :]), axis = 1)

        data_x_not_expanded = data_transform[:, :-1]
        data_y = data_transform[:, [-1]]

        expand_data_obj = ExpandData(data, train_idx, test_idx, sliding, expand_func = expand_func)
        data_expanded = expand_data_obj.process_data()

        data_X = np.concatenate((data_x_not_expanded, data_expanded), axis = 1)

        self.X_train, self.X_test, self.y_train, self.y_test = data_X[:train_idx, :], data_X[train_idx:, :], data_y[:train_idx, :], data_y[train_idx:, :]

    def train(self, epochs = 2000):
        self.preprocessing_data()

        p = Population(self.pop_size, self.c1, self.c2, activation = self.activation)

        best = p.train(self.X_train, self.y_train)

        pred = best.predict(self.X_test) 

        self.pred_inverse = self.scaler.inverse_transform(pred)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        print(mean_absolute_error(self.real_inverse, self.pred_inverse))

        self.draw_predict()

model = Model(dt.values, train_idx, test_idx, sliding)
model.train()