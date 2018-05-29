import numpy as np
import random
import copy
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class FLNN:
    def __init__(self, dataset_original, train_idx, test_idx, sliding, activation = 0, data_filename = "3_minutes", test = 'tn1', e_func = 0, learning_rate = 0.05, batch_size = 16, beta = 0.9):
        self.dataset_original = dataset_original[:test_idx+sliding, :]
        self.sliding = sliding
        self.method_statistic = 0
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta = beta
        self.min_max_scaler = MinMaxScaler()
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.dimension = dataset_original.shape[1]
        self.n_expanded = 2
        self.e_func = e_func
        self.pathsave = "test/" + test + "/"
        self.textfilename = test
        self.filename = "{0}-flnn_sliding_{1}-activation_{2}-e_func_{3}".format(data_filename, sliding, activation, e_func)
    
    def save_file_csv(self):
        t1 = np.concatenate( (self.y_test_inverse, self.y_pred_inverse), axis = 1)
        np.savetxt(self.pathsave + self.filename + ".csv", t1, delimiter=",")

    def predict(self):
        Z = np.dot(self.W, self.X_test) + self.b
        y_pred = self.tanh(Z)

        self.y_pred_inverse = self.inverse_data(y_pred).T
        self.y_test_inverse = self.inverse_data(self.y_test).T

        self.score_test_MAE = mean_absolute_error(self.y_pred_inverse, self.y_test_inverse)
        self.score_test_RMSE = np.sqrt(mean_squared_error(self.y_pred_inverse, self.y_test_inverse))

        self.write_to_result_file()

        self.draw_predict()
        # self.save_file_csv()

    def write_to_result_file(self):
        with open(self.pathsave + self.textfilename + '.txt', 'a') as file:
            file.write("{0}  -  {1} - {2}\n".format(self.filename, self.score_test_MAE, self.score_test_RMSE))

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.y_test_inverse[:, 0][0:200], color='#009FFD', linewidth=2.5)
        plt.plot(self.y_pred_inverse[:, 0][0:200], color='#FFA400', linewidth=2.5)
        plt.ylabel('CPU')
        plt.xlabel('Timestamp')
        plt.legend(['Actual', 'Prediction'], loc='upper right')
        plt.savefig(self.pathsave + self.filename + ".png")
        # plt.show()
        plt.close()

    def inverse_data(self, transform_data):
        self.min_max_scaler.fit_transform(self.dataset_original[:, [0]])
        
        return self.min_max_scaler.inverse_transform(transform_data)

    def power_polynomials(self, n = 2):
        expanded_results = np.zeros((self.dataset_original.shape[0], 1))
        
        for i in range(self.dimension):
            for j in range(2, n+2):
                expanded = np.power(self.dataset_original[:, [i]], j)
                
                expanded_results = np.concatenate((expanded_results, expanded), axis = 1)
        
        expanded_results = expanded_results[:, 1:]
    
        return expanded_results
    
    def chebyshev_polynomials(self, n = 2):
        expanded_results = np.zeros((self.dataset_original.shape[0], 1))
    
        for i in range(self.dimension):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n+2):
                c = 2 * self.dataset_original[:, [i]] * c2 - c1
                c1 = c2
                c2 = c
    
                expanded_results = np.concatenate((expanded_results, c), axis=1)
    
        return expanded_results[:, 1:]

    def legendre_data(self, n = 2):
        expanded = np.zeros((self.dataset_original.shape[0], 1))

        for i in range(self.dimension):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n+2):
                c = ((2 * j + 1) * self.dataset_original[:, [i]] * c2 - j * c1) / (j + 1)
                c1 = c2
                c2 = c

                expanded = np.concatenate((expanded, c), axis=1)

        return expanded[:, 1:]

    def laguerre_data(self, n = 2):
        expanded = np.zeros((self.dataset_original.shape[0], 1))

        for i in range(self.dimension):
            c1 = np.ones((self.dataset_original.shape[0], 1))
            c2 = self.dataset_original[:, [i]]
            for j in range(2, n + 2):
                c = ((2 * j + 1 - self.dataset_original[:, [i]]) * c2 - j * c1) / (j + 1)
                c1 = c2
                c2 = c

                expanded = np.concatenate((expanded, c), axis=1)

        return expanded[:, 1:]
        
    def processing_data_2(self):
        dataset_original, train_idx, test_idx, sliding, n_expanded, method_statistic = self.dataset_original, self.train_idx, self.test_idx , self.sliding, self.n_expanded, self.method_statistic
        
        list_split = []        
        for i in range(self.dimension):
            list_split.append(dataset_original[:, i:i+1])
        
        # Expanded
        expanded = None
        if self.e_func == 0:
            expanded = self.chebyshev_polynomials(n_expanded)
        elif self.e_func == 1:
            expanded = self.legendre_data(n_expanded)
        elif self.e_func == 2:
            expanded = self.laguerre_data(n_expanded)
        elif self.e_func == 3:
            expanded = self.power_polynomials(n_expanded)
        for i in range(expanded.shape[1]):
            list_split.append(expanded[:, i:i+1])
        
        list_transform = []
        for i in range(len(list_split)):
            list_transform.append(self.min_max_scaler.fit_transform(list_split[i]))
            
        features = len(list_transform)
        
        dataset_sliding = np.zeros((test_idx, 1))
        for i in range(len(list_transform)):
            for j in range(sliding):
                d = np.array(list_transform[i][j:test_idx + j])
                dataset_sliding = np.concatenate((dataset_sliding, d), axis = 1)
        dataset_sliding = dataset_sliding[:, 1:]
        
        dataset_y = copy.deepcopy(list_transform[0][sliding:]) 
        
        if method_statistic == 0:
            dataset_X = copy.deepcopy(dataset_sliding)
        elif method_statistic == 1:
            dataset_X = np.zeros((dataset_sliding.shape[0], 1))
            for i in range(features):    
                mean = np.reshape(np.mean(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                dataset_X = np.concatenate((dataset_X, mean), axis = 1)
            dataset_X = dataset_X[:, 1:]
        elif method_statistic == 2:
            dataset_X = np.zeros((dataset_sliding.shape[0], 1))
            for i in range(features):    
                min_X = np.reshape(np.amin(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                median_X = np.reshape(np.median(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                max_X = np.reshape(np.amax(dataset_sliding[:, i*sliding:i*sliding + sliding], axis = 1), (-1, 1))
                dataset_X = np.concatenate((dataset_X, min_X, median_X, max_X), axis = 1)
            dataset_X = dataset_X[:, 1:]     
        
        # train_size = int(dataset_X.shape[0] * 0.8)
        X_train, y_train, X_test, y_test = dataset_X[:train_idx, :], dataset_y[:train_idx, :], dataset_X[train_idx:, :], dataset_y[train_idx:, :]

        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.T
        y_test = y_test.T

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def init_parameters(self, n_inputs, n_outputs):
        W = np.random.randn(n_outputs, n_inputs)
        b = np.zeros((n_outputs, 1))
        
        return W, b

    def init_momentum_parameters(self, n_inputs, n_outputs):
        vdW = np.zeros((n_outputs, n_inputs))
        vdb = np.zeros((n_outputs, 1))
        
        return vdW, vdb

    def tanh(self, x):
        e_plus = np.exp(x)
        e_minus = np.exp(-x)
        
        return (e_plus - e_minus) / (e_plus + e_minus)

    def tanh_backward(self, x):
        return 1 - np.square(x)

    def random_mini_batches(self, seed = 0):
        X, Y = self.X_train, self.y_train
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

    def train(self, epochs = 300):
    
        self.processing_data_2()

        X_train, y_train, X_test, y_test = self.X_train, self.y_train, self.X_test, self.y_test

        seed = 1
        
        n_inputs = X_train.shape[0]
        n_outputs = y_train.shape[0]
        
        W, b = self.init_parameters(n_inputs, n_outputs)
        
        vdW, vdb = self.init_momentum_parameters(n_inputs, n_outputs)
        
        for e in range(epochs):
            
            seed += 1
            
            mini_batches = self.random_mini_batches(seed = seed)
            
            total_cost = 0
            
            for mini_batch in mini_batches:
                X_batch, y_batch = mini_batch
                
                m = X_batch.shape[1]
                
                # Forward
                Z = np.dot(W, X_batch) + b
                A = self.tanh(Z)

                # Backpropagation
                dA = A - y_batch
                dZ = dA * self.tanh_backward(A)

                db = 1./m * np.sum(dZ, axis = 1, keepdims=True)
                dW = 1./m * np.dot(dZ, X_batch.T)
                
                vdW = self.beta * vdW + (1 - self.beta) * dW
                vdb = self.beta * vdb + (1 - self.beta) * db

                W -= self.learning_rate * vdW
                b -= self.learning_rate * vdb

            Z_t = np.dot(W, X_train) + b
            A_t = self.tanh(Z_t)
                
            mae_train = mean_absolute_error(A_t, y_train)
            # print("MAE Train: %.5f" % (mae_train))

        self.W = W
        self.b = b

        self.predict()
     