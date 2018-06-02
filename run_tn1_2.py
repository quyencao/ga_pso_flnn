from ga_flnn.model import Model as GAModel
from pso_flnn.model import Model as PSOModel
from flnn.model import Model as FLNNModel
import pandas as pd

filename1 = 'data_resource_usage_3Minutes_6176858948.csv'
filename2 = 'data_resource_usage_5Minutes_6176858948.csv'
filename3 = 'data_resource_usage_8Minutes_6176858948.csv'
filename4 = 'data_resource_usage_10Minutes_6176858948.csv'

filenames = [filename1, filename2, filename3, filename4]
fses = ['3_minutes', '5_minutes','8_minutes', '10_minutes']

foldername = 'data/'

# parameters
list_idx = [(10560, 2640), (6640, 1660), (4160, 1040), (3280, 820)]
sliding_windows = [2, 3, 4, 5, 6]
methods = ['PSO']

for index, filename in enumerate(filenames):

    df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3])
    df.dropna(inplace=True)
    dataset_original = df.values

    f = fses[index]

    idx = list_idx[index]

    for method in methods:
        for sw in sliding_windows:
            if method == 'GA':
                p = GAModel(dataset_original, idx[0], idx[1], sw, expand_func=0, pop_size=100, pc=0.8, pm=0.02,
                            activation=1, data_filename= f, test = "tn1")
                p.train(epochs=1000)
            elif method == 'PSO':
                p = PSOModel(dataset_original, idx[0], idx[1], sw, expand_func=0, pop_size=100, c1=1.2, c2=1.2,
                            activation=1, data_filename=f, test="tn1")
                p.train(epochs=1000)
            elif method == 'FLNN':
                p = FLNNModel(dataset_original, idx[0], idx[1], sw, expand_func = 0, activation = 1, data_filename=f, test="tn1")
                p.train(epochs=1000)

