import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Note: All structures should be numpy arrays for rnn

def standardize_column(column_data):
    column_data = column_data.reshape(-1,1)
    sc = StandardScaler()
    column_data = sc.fit_transform(column_data)
    return column_data
def separate_data(column_data,index):
    column_data = column_data.tolist()
    
    isolated_data = []
    for i in range(len(column_data)):
        isolated_data.append(column_data[i][index])
    isolated_data = np.array(isolated_data)
    return isolated_data

#import nasdaq data
nasdaq_data = pd.read_csv('NASDAQ_index.csv')
nasdaq_data = nasdaq_data.iloc[:1258,1:2].values
#import training set
google_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
google_open = google_dataset.iloc[:,1:2].values

data = {'NASDAQ open':nasdaq_data.flatten(),'google_open':google_open.flatten()}
training_set = pd.DataFrame(data = data)

cols = list(training_set)[0:3]
training_set = training_set[cols].astype(str)
for i in cols:
    for j in range(0,len(training_set)):
        training_set[i][j] = training_set[i][j].replace(",","")
        
training_set = training_set.astype(float)
training_set = training_set.values
#separate out the columns and standardize
training_set_first_column = separate_data(training_set,0)
training_set_first_column = standardize_column(training_set_first_column)
training_set_second_column = separate_data(training_set,1)
training_set_second_column = standardize_column(training_set_second_column)

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
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences=True))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences=True))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences=False))
#regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1, activation='linear'))
#rmsprop recommended for rnns according to keras documentation
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
regressor.fit(x_train,y_train,epochs=100,batch_size=32)

test_set = pd.read_csv('Google_Stock_Price_Test.csv')
actual_price_2017 = test_set.iloc[:,1:2].values

#get predications for the 2017 test set
# for each prediction, we are using the 60 previous timestamps
#this will involve concatenation of the test and training set

#include NASDAQ data in training set 
google_dataset['NASDAQ Open'] = nasdaq_data
#include nasdaq data relevant for the test set
all_nasdaq = pd.read_csv("NASDAQ_index.csv")
test_set["NASDAQ Open"] = all_nasdaq.iloc[1258:,1:2].values
#combine train and test-trying out Date for verifying
#dataset_total = pd.concat((google_dataset[['Date','Open','NASDAQ open']],
#                           test_set[['Date','Open','NASDAQ open']]),axis=0
#                          ,ignore_index=True)
dataset_total = pd.concat((google_dataset[['Open','NASDAQ Open']],
                           test_set[['Open','NASDAQ Open']]),axis=0,
                          ignore_index=True)
inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values
inputs = scalar.transform(inputs)

x_test = []
#for each timestamp to predict, append the 60 previous timestamps
for i in range(60,80):
    x_test.append(inputs[i-60:i,0:2])
x_test =np.array(x_test)
predicted_stock_price = regressor.predict(x_test)

#convert to list and back to change the dimension to (20,2)
#convert from np array to list and back again to remove dummy elements
predicted_stock_price = predicted_stock_price.tolist()
for i in range(20):
    predicted_stock_price[i].append(0)
predicted_stock_price = np.array(predicted_stock_price)
#predicted_stock_price = scalar.inverse_transform(predicted_stock_price)
predicted_stock_price = predicted_stock_price.tolist()
for i in range(20):
    predicted_stock_price[i] = predicted_stock_price[i][0:1]
predicted_stock_price = np.array(predicted_stock_price)

actual_price_2017 = actual_price_2017.tolist()
for i in range(20):
    #print(actual_price_2017[i])
    actual_price_2017[i].append(0)
actual_price_2017 = np.array(actual_price_2017)
actual_price_2017 = scalar.fit_transform(actual_price_2017)
actual_price_2017 = actual_price_2017.tolist()
for i in range(20):
    actual_price_2017[i] = actual_price_2017[i][0:1]
actual_price_2017 = np.array(actual_price_2017)

def standardize_column(data):
    data = data.reshape(-1,1)
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data
def separate_data(data,index):
    isolated_data = []
    for i in range(len(data)):
        isolated_data.append(data[i][index])
    isolated_data = np.array(isolated_data)
    return isolated_data
        

plt.plot(actual_price_2017)
plt.title("Actual vs predicted")
plt.plot(predicted_stock_price)












    
