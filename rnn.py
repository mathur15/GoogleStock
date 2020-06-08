import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#import training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#instead of standardising, normalize for rnn
scalar = MinMaxScaler()
training_set_scaled = scalar.fit_transform(training_set)
