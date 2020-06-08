import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = np.array(dataset_train.iloc[:,1])
#alternate way to convert to numpy arrays
#training_set = dataset_train.iloc[:,1:2].values