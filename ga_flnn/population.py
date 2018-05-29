import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt
from chormesome import Chormesome

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

        for e in range(epochs):
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

            print("> Epoch {0}: Best fitness {1}".format(e + 1, best_fitness))
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

                for i in range(d+1):
                    if random.uniform(0, 1) < self.pm:
                        w1 = c1.w
                        idx = np.random.randint(d+1)
                        w1[idx] = np.random.uniform(low=-1, high=1)

                        c1 = Chormesome(d, w1)
                for i in range(d+1):
                    if random.uniform(0, 1) < self.pm:
                        w2 = c2.w
                        idx = np.random.randint(d + 1)
                        w2[idx] = np.random.uniform(low=-1, high=1)

                        c2 = Chormesome(d, w2)
                self.next_population.append(c1)
                self.next_population.append(c2)
            self.population = self.next_population

        return best_chormesome