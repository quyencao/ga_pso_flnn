import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt
from expand_data import ExpandData
from population import Population

class Model:
    def __init__(self, data_original, train_idx, test_idx, sliding, expand_func = 0, pop_size = 100, pc = 0.7, pm = 0.01,
                 activation = 0, data_filename= "3_minutes", test = "tn1"):
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
        self.pathsave = "test/" + test + "/"
        self.textfilename = test
        self.filename = "{0}-GA-FLNN-sliding_{1}-expand_func_{2}-pop_size_{3}-pc_{4}-pm_{5}-activation_{6}".format(data_filename, sliding, expand_func, pop_size, pc, pm, activation)

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.real_inverse[:, 0][:200], color='#009FFD', linewidth=2.5)
        plt.plot(self.pred_inverse[:, 0][:200], color='#FFA400', linewidth=2.5)
        plt.ylabel('CPU')
        plt.xlabel('Timestamp')
        plt.legend(['Actual', 'Prediction'], loc='upper right')
        plt.savefig(self.pathsave + self.filename + ".png")
        # plt.show()
        plt.close()

    def write_to_result_file(self):
        with open(self.pathsave + self.textfilename + '.txt', 'a') as file:
            file.write("{0}  -  {1}  -  {2}\n".format(self.filename, self.mae, self.rmse))
    
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

        best = p.train(self.X_train, self.y_train, epochs=epochs)

        pred = best.predict(self.X_test) 

        self.pred_inverse = self.scaler.inverse_transform(pred)
        self.real_inverse = self.scaler.inverse_transform(self.y_test)

        self.mae = mean_absolute_error(self.real_inverse, self.pred_inverse)
        self.rmse = np.sqrt(mean_squared_error(self.real_inverse, self.pred_inverse))

        print(self.mae)

        self.draw_predict()

        self.write_to_result_file()