import pandas as pd

from func import init_hamming, init_hopfield

train_sample = pd.read_excel('examples.xlsx', usecols="B:I", names=[0, 1, 2, 3, 4, 5, 6, 7], nrows=32,
                             engine='openpyxl')
test_sample = pd.read_excel('examples.xlsx', usecols="B:I", names=[0, 1, 2, 3, 4, 5, 6, 7], skiprows=32,
                            engine='openpyxl')
n_col = train_sample.shape[1]
n_row = int(train_sample.shape[0] / 4)

init_hopfield(train=train_sample, test=test_sample, n_stand=4, n_col=n_col, n_row=n_row)
init_hamming(train_sample, test_sample)
