from ga_flnn.model import Model as GAModel
from pso_flnn.model import Model as PSOModel
from flnn.flnn import FLNN
import numpy as np
import pandas as pd

filename1 = 'data_resource_usage_3Minutes_6176858948.csv'
filename2 = 'data_resource_usage_5Minutes_6176858948.csv'
filename3 = 'data_resource_usage_8Minutes_6176858948.csv'
filename4 = 'data_resource_usage_10Minutes_6176858948.csv'

filenames = [filename1]
fses = ['3m']

foldername = 'data/'

# parameters
list_idx = [(11120, 2780)]
cross_rates = [0.7, 0.75, 0.8, 0.85, 0.9]
mutate_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
c1s = [0.7, 1.2, 1.5, 1.7, 2.0]
c2s = [0.7, 1.2, 1.5, 1.7, 2.0]
sliding_windows = [4]
pop_sizes = [100]
method_statistic = [0]
methods = ['GA', 'PSO']
activations = [1]
expanded_functions = [0, 1, 2, 3]

for index, filename in enumerate(filenames):

    df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3])
    df.dropna(inplace=True)
    dataset_original = df.values

    f = fses[index]

    idx = list_idx[index]

    for method in methods:
        if method == 'GA':
            for ef in expanded_functions:
                for pop_size in pop_sizes:
                    for cr in cross_rates:
                        for mr in mutate_rates:
                            p = GAModel(dataset_original, idx[0], idx[1], sliding_windows[0],
                                        expand_func=ef, pop_size=pop_size, pc=cr, pm=mr,
                                        activation=1, data_filename=f, test="tn3")
                            p.train(epochs=1000)

        if method == 'PSO':
            for ef in expanded_functions:
                for pop_size in pop_sizes:
                    for c1 in c1s:
                        for c2 in c2s:
                            p = PSOModel(dataset_original, idx[0], idx[1], sliding_windows[0],
                                         expand_func=ef, pop_size=pop_size, c1=c1, c2=c2,
                                         activation=1, data_filename=f, test="tn3")
                            p.train(epochs=1000)

