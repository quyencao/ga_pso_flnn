import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt

dt = pd.read_csv('data_resource_usage_8Minutes_6176858948.csv', usecols=[3], names=['cpu'])

sliding = 2
train_idx = 4420
test_idx = 780

# data = dt.values[:train_idx + test_idx + sliding + 1, :]

# scaler = MinMaxScaler()
# data_scale = scaler.fit_transform(data)

# data_transform = data_scale[:train_idx + test_idx, :]

# for i in range(1, sliding+1):
#     data_transform = np.concatenate((data_transform, data_scale[i:i+train_idx + test_idx, :]), axis = 1)

# data_X = data_transform[:, :-1]
# data_y = data_transform[:, [-1]]

# scaler2 = MinMaxScaler()
# x2 = 2 * np.power(data, 2) - 1
# x3 = 4 * np.power(data, 3) - 3 * data
# x4 = 8 * np.power(data, 4) - 8 * np.power(data, 2) + 1
# x5 = 2 * data * x4 - x3
# x6 = 2 * data * x5 - x4

# x2 = scaler2.fit_transform(x2)
# x3 = scaler2.fit_transform(x3)
# x4 = scaler2.fit_transform(x4)
# x5 = scaler2.fit_transform(x5)
# x6 = scaler2.fit_transform(x6)

# data_expanded = np.ones(data_scale[:train_idx + test_idx, :].shape)

# for i in range(0, sliding):
#     data_expanded = np.concatenate((data_expanded, x2[i:i + train_idx + test_idx, :]), axis=1)

# for i in range(0, sliding):
#     data_expanded = np.concatenate((data_expanded, x3[i:i + train_idx + test_idx, :]), axis=1)

# for i in range(0, sliding):
#     data_expanded = np.concatenate((data_expanded, x4[i:i + train_idx + test_idx, :]), axis=1)

# for i in range(0, sliding):
#     data_expanded = np.concatenate((data_expanded, x5[i:i + train_idx + test_idx, :]), axis=1)

# for i in range(0, sliding):
#     data_expanded = np.concatenate((data_expanded, x6[i:i + train_idx + test_idx, :]), axis=1)

# data_expanded = data_expanded[:, 1:]

# data_x_final = np.concatenate((data_X, data_expanded), axis = 1)

# data_x_train, data_x_test, data_y_train, data_y_test = data_x_final[:train_idx, :], data_x_final[train_idx:, :], data_y[:train_idx, :], data_y[train_idx:, :]

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

class Chormesome:
    def __init__(self, n, w = None, activation = 0):
        self.w = np.random.uniform(low = -5, high = 5, size=(n, 1))

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
        z = np.dot(X, self.w)
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
        z = np.dot(X, self.w)
        a = self.activation_output(z)
        return a

class Population:
    def __init__(self, n, pc, pm, activation = 0):
        self.n = n
        self.pc = pc
        self.pm = pm
        self.activation = activation

    def init_pop(self, d):
        self.population = []
        for i in range(0, self.n):
            c = Chormesome(d, activation = self.activation)
            self.population.append(c)

    def get_index(self, fitnesses, sum_fitness):
        r = np.random.uniform(low=0, high=sum_fitness)
        for idx, f in enumerate(fitnesses):
            r -= f
            if r < 0:
                return idx

    def train(self, X, y, epochs=2000):
        best_fitness = -1
        best_chormesome = None

        d = X.shape[1]

        self.init_pop(d)

        for i in range(epochs):
            # Tinh Fitness
            fitnesses = []

            for p in self.population:
                fitness = p.compute_fitness(X, y)
                fitnesses.append(fitness)

                # print(fitness)

            fitnesses = np.array(fitnesses)

            sort_index = np.argsort(fitnesses)

            if fitnesses[sort_index[-1]] > best_fitness:
                best_fitness = fitnesses[sort_index[-1]]
                best_chormesome = copy.deepcopy(self.population[sort_index[-1]])

            print(best_fitness)
            # Produce
            self.next_population = []

            sum_fitness = np.sum(fitnesses)

            while (len(self.next_population) < self.n):
                c1 = self.population[self.get_index(fitnesses, sum_fitness)]
                c2 = self.population[self.get_index(fitnesses, sum_fitness)]

                if random.uniform(0, 1) < self.pc:
                    w1 = 0.7 * c1.w + 0.3 * c2.w
                    w2 = 0.3 * c2.w + 0.7 * c1.w

                    c1 = Chormesome(d, w1)
                    c2 = Chormesome(d, w2)

                for i in range(d):
                    if random.uniform(0, 1) < self.pm:
                        w1 = c1.w
                        idx = np.random.randint(d)
                        w1[idx] = np.random.uniform(low=-3, high=3)

                        c1 = Chormesome(d, w1)
                for i in range(d):
                    if random.uniform(0, 1) < self.pm:
                        w2 = c2.w
                        idx = np.random.randint(d)
                        w2[idx] = np.random.uniform(low=-3, high=3)

                        c2 = Chormesome(d, w2)
                self.next_population.append(c1)
                self.next_population.append(c2)
            self.population = self.next_population

        return best_chormesome

class Model:
    def __init__(self, data_original, train_idx, test_idx, sliding, expand_func = 0, pop_size = 100, pc = 0.7, pm = 0.01, activation = 0):
        self.data_original = data_original
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.data = data_original[:train_idx + test_idx + sliding + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.activation = activation

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.real_inverse[:, 0][:200], color='#009FFD', linewidth=2)
        plt.plot(self.pred_inverse[:, 0][:200], color='#FFA400', linewidth=2)
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

    def train(self, epochs = 1000):
        self.preprocessing_data()

        p = Population(self.pop_size, self.pc, self.pm, activation = self.activation)

        best = p.train(self.X_train, self.y_train)

        pred = best.predict(self.X_test) 

        self.pred_inverse = self.scaler.inverse_transform(pred)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        

        print(mean_absolute_error(self.real_inverse, self.pred_inverse))

        self.draw_predict()

# p = Population(100)

# best = p.train(data_x_train, data_y_train)

# mae, pred, real = best.compute_mae(data_x_test, data_y_test)

# print(mae)

model = Model(dt.values, train_idx, test_idx, sliding)
model.train()