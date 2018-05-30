from ga_flnn.model import Model as GAModel
import pandas as pd

filename1 = 'data_resource_usage_3Minutes_6176858948.csv'
filename2 = 'data_resource_usage_5Minutes_6176858948.csv'
filename3 = 'data_resource_usage_8Minutes_6176858948.csv'
filename4 = 'data_resource_usage_10Minutes_6176858948.csv'

filenames = [filename4]
fses = ['5_minutes']

foldername = 'data/'

# parameters
list_idx = [(3280, 820)]
sliding_windows = [2, 3, 5]
pop_sizes = [100]
cross_rates = [0.7, 0.75, 0.8, 0.85, 0.9]
mutate_rates = [0.01, 0.02, 0.03, 0.04, 0.05]

"""
    RUN GA - FLNN
"""
for index, filename in enumerate(filenames):

    df = pd.read_csv(foldername + filename, header=None, index_col=False, usecols=[3])
    df.dropna(inplace=True)
    dataset_original = df.values

    f = fses[index]

    idx = list_idx[index]

    for pop_size in pop_sizes:
        for cr in cross_rates:
            for mr in mutate_rates:
                p = GAModel(dataset_original, idx[0], idx[1], sliding_windows[0],
                            expand_func=0, pop_size=pop_size, pc=cr, pm=mr,
                            activation=1, data_filename=f, test="tn5")
                p.train(epochs=1000)
