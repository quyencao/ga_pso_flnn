from pso_flnn.model import Model as PSOModel
import pandas as pd

filename1 = 'data_resource_usage_3Minutes_6176858948.csv'
filename2 = 'data_resource_usage_5Minutes_6176858948.csv'
filename3 = 'data_resource_usage_8Minutes_6176858948.csv'
filename4 = 'data_resource_usage_10Minutes_6176858948.csv'

filenames = [filename1]
fses = ['3_minutes']

foldername = 'data/'

# parameters
list_idx = [(11120, 2780)]
c1s = [0.7, 1.2, 1.5, 1.7, 2]
c2s = [0.7, 1.2, 1.5, 1.7, 2]
sliding_windows = [2, 3, 4, 5, 6, 7, 8, 9, 10]
pop_sizes = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

for index, filename in enumerate(filenames):

    df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3])
    df.dropna(inplace=True)
    dataset_original = df.values

    f = fses[index]

    idx = list_idx[index]

    for pop_size in pop_sizes:
        for c1 in c1s:
            for c2 in c2s:
                p = PSOModel(dataset_original, idx[0], idx[1], sliding_windows[0],
                             expand_func=0, pop_size=pop_size, c1=c1, c2=c2,
                             activation=1, data_filename=f, test="tn4")
                p.train(epochs=1000)

