import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Note: All structures should be numpy arrays for rnn

#import training set
#nasdaq_dataset = pd.read_csv('NASDAQ_index.csv')
#nasdaq_open = nasdaq_dataset['Open']
#nasdaq_open_train = nasdaq_open.iloc[:1258]

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#training_set_new = pd.concat([nasdaq_open_train,dataset_train['Open']],axis=1)
training_set_new = dataset_train.iloc[:,1:2].values

#instead of standardising, normalize for rnn
scalar = MinMaxScaler()
training_set_scaled = scalar.fit_transform(training_set_new)

#trends based on 60 previous timestamps and 1 output(time t+1)
#for each observation to predict 60 preious financial days will be used
#y_train contain the actual price
#each row in x_train correspond to the 60 previous timestamps for a given day
x_train = []
y_train = []
#x_train_open = []
#x_train_nasdaq = []
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    #x_train_nasdaq.append(training_set_scaled[i,1])
    y_train.append(training_set_scaled[i,0])
#x_train.append([x_train_open])
#x_train.append([x_train_nasdaq])
#x_train_open = np.array(x_train_open)
#x_train_nasdaq = np.array(x_train_nasdaq)
#x_train_nasdaq = x_train_nasdaq.reshape(-1,1)
#x_train = np.concatenate(x_train_open,x_train_nasdaq)
x_train,y_train = np.array(x_train),np.array(y_train)

#add more indicators- convert x_train to 3 dimensional for rnn layers
#so far only one indicator---open stock price
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

regressor = Sequential()

regressor.add(LSTM(units = 50,return_sequences=True,
                   input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

#rmsprop recommended for rnns according to keras documentation
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

regressor.fit(x_train,y_train,epochs=100,batch_size=32)
    
