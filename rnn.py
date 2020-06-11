import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Note: All structures should be numpy arrays for rnn

#import nasdaq data
nasdaq_data = pd.read_csv('NASDAQ_index.csv')
nasdaq_data = nasdaq_data.iloc[:1258,1:2].values

#import training set
google_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
google_open = google_dataset.iloc[:,1:2].values

google_dataset_predict = pd.read_csv('Google_Stock_Price_Test.csv')

data = {'NASDAQ open':nasdaq_data.flatten(),'google_open':google_open.flatten()}

training_set = pd.DataFrame(data = data)
cols = list(training_set)[0:3]

training_set = training_set[cols].astype(str)
for i in cols:
    for j in range(0,len(training_set)):
        training_set[i][j] = training_set[i][j].replace(",","")
 
training_set = training_set.astype(float)
training_train = training_set.values

#instead of standardising, normalize for rnn
scalar = MinMaxScaler()
training_set_scaled = scalar.fit_transform(training_set)

#trends based on 60 previous timestamps and 1 output(time t+1)
#for each observation to predict 60 preious financial days will be used
#y_train contain the actual price
#each row in x_train correspond to the 60 previous timestamps for a given day
x_train = []
y_train = []

for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0:2])
    y_train.append(training_set_scaled[i,0])

x_train,y_train = np.array(x_train),np.array(y_train)
#print(x_train.shape[0])
#print(x_train.shape[1])

regressor = Sequential()

regressor.add(LSTM(units = 50,return_sequences=True,
                   input_shape=(x_train.shape[1],2)))
#for training each data point there is 60 rows and 2 columns of data
#see x_train to understand how the data is prepared
#input shape- think of it as the data needed for one row/datapoint
#input_shape=(timesteps,features)

regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50,return_sequences=False))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1, activation='linear'))

#rmsprop recommended for rnns according to keras documentation
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

regressor.fit(x_train,y_train,epochs=200,batch_size=32)



    
