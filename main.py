import pandas as pd

from func import start_hamming, start_hopfield

train_s = pd.read_excel('traintest.xlsx', usecols="B:I", names=[0, 1, 2, 3, 4, 5, 6, 7], nrows=32,
                        engine='openpyxl')
test_s = pd.read_excel('traintest.xlsx', usecols="B:I", names=[0, 1, 2, 3, 4, 5, 6, 7], skiprows=32,
                       engine='openpyxl')
cols = train_s.shape[1]
rows = int(train_s.shape[0] / 4)

start_hopfield(train=train_s, test=test_s, cols=cols, rows=rows)
start_hamming(train_s, test_s)
