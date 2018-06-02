import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt
from expand_data import ExpandData
import math

class Model:
    def __init__(self, data_original, train_idx, test_idx, sliding, expand_func = 0,
                 activation = 2, lr = 0.05, batch_size = 32, beta = 0.9, data_filename= "3_minutes", test = "flnn"):
        self.data_original = data_original
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.sliding = sliding
        self.data = data_original[:train_idx + test_idx + sliding + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.activation = activation
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.textfilename = test
        self.pathsave = "test/" + test + "/"
        self.filename = "{0}-FLNN-sliding_{1}-expand_func_{2}-activation_{3}_batchsize_{4}".format(data_filename, sliding, expand_func, activation, batch_size)

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

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.real_inverse[:, 0][0:200], color='#009FFD', linewidth=2.5)
        plt.plot(self.pred_inverse[:, 0][0:200], color='#FFA400', linewidth=2.5)
        plt.ylabel('CPU')
        plt.xlabel('Timestamp')
        plt.legend(['Actual', 'Prediction'], loc='upper right')
        plt.savefig(self.pathsave + self.filename + ".png")
        # plt.show()
        plt.close()

    def write_to_result_file(self):
        with open(self.pathsave + self.textfilename + '.txt', 'a') as file:
            file.write("{0}  -  {1}  -  {2}\n".format(self.filename, self.mae, self.rmse))

    def tanh(self, x):
        exp_plus = np.exp(x)
        exp_minus = np.exp(-x)

        return (exp_plus - exp_minus) / (exp_plus + exp_minus)

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def elu(self, x, alpha = 0.9):
        return np.where(x < 0, alpha * (np.exp(x) - 1), x)    

    def save_file_csv(self):
        t1 = np.concatenate( (self.pred_inverse, self.real_inverse), axis = 1)
        np.savetxt(self.pathsave + self.filename + '.csv', t1, delimiter=",") 

    def write_to_result_file(self):
        with open(self.pathsave + self.textfilename + '.txt', 'a') as file:
            file.write("{0}  -  {1}  -  {2}\n".format(self.filename, self.mae, self.rmse))

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

    def activation_backward(self, a):
        if self.activation == 0:
            return 1
        elif self.activation == 1:
            return 1 - np.power(a, 2)
        elif self.activation == 2:
            return np.where(a < 0, 0, 1)
        elif self.activation == 3:
            return np.where(a < 0, a + 0.9, 1)    

    def random_mini_batches(self, seed = 0):
        X, Y = self.X_train.T, self.y_train.T
        mini_batch_size = self.batch_size
        
        m = X.shape[1]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)
        
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def init_parameters(self, d):   
        return np.random.randn(d, 1), np.zeros((1, 1))   

    def init_momentum_parameters(self, d):
        vdw = np.zeros((d, 1))
        vdb = np.zeros((1, 1))
        
        return vdw, vdb

    def train(self, epochs = 1000):
        self.preprocessing_data()

        d = self.X_train.shape[1]

        seed = 0

        w, b = self.init_parameters(d)

        vdw, vdb = self.init_momentum_parameters(d)

        for e in range(epochs):

            seed += 1

            mini_batches = self.random_mini_batches(seed = seed)

            total_error = 0

            for mini_batch in mini_batches:
                X_batch, y_batch = mini_batch

                X_batch = X_batch.T
                y_batch = y_batch.T

                m = X_batch.shape[0]

                # Feed Forward
                z = np.dot(X_batch, w) + b
                a = self.activation_output(z)

                total_error += mean_absolute_error(a, y_batch)

                # Backpropagation
                da = a - y_batch
                dz = da * self.activation_backward(a)

                db = 1./m * np.sum(dz, axis = 0, keepdims=True)
                dw = 1./m * np.matmul(X_batch.T, dz)

                vdw = self.beta * vdw + (1 - self.beta) * dw
                vdb = self.beta * vdb + (1 - self.beta) * db

                # Update weights
                w -= self.lr * vdw
                b -= self.lr * vdb
            
            print("> Epoch {0}: MAE {1}".format(e, total_error))


        z = np.dot(self.X_test, w) + b
        a = self.activation_output(z)
        
        self.pred_inverse = self.scaler.inverse_transform(a)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = mean_absolute_error(self.pred_inverse, self.real_inverse)
        self.rmse = np.sqrt(mean_squared_error(self.pred_inverse, self.real_inverse))

        self.save_file_csv()
        self.write_to_result_file()

        self.draw_predict()


# dt = pd.read_csv('data_resource_usage_10Minutes_6176858948.csv', usecols=[3], names=['cpu'])

# sliding = 2
# train_idx = 3280
# test_idx = 820

# model = Model(dt.values, train_idx, test_idx, sliding)
# model.train()