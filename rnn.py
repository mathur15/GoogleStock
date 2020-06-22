import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#Note: All structures should be numpy arrays for rnn

sc = StandardScaler()
def standardize_column(column_data):
    sc = StandardScaler()
    column_data = column_data.reshape(-1,1)
    column_data = sc.fit_transform(column_data)
    return column_data

#import nasdaq data
nasdaq_data = pd.read_csv('NASDAQ_index.csv')
nasdaq_data = nasdaq_data.iloc[:1258,1:2].values
#import training set
google_dataset = pd.read_csv('Google_Stock_Price_Train.csv')
google_open = google_dataset.iloc[:,1:2].values

#separate out the columns and standardize
training_set_first_column = standardize_column(nasdaq_data)
training_set_second_column = standardize_column(google_open)

data = {'NASDAQ open':training_set_first_column.flatten(),
        'google_open':training_set_second_column.flatten()}
training_set = pd.DataFrame(data = data)

cols = list(training_set)[0:3]
training_set = training_set[cols].astype(str)
for i in cols:
    for j in range(0,len(training_set)):
       training_set[i][j] = training_set[i][j].replace(",","")
training_set = training_set.astype(float)
training_set = training_set.values        

#trends based on 60 previous timestamps and 1 output(time t+1)
#for each observation to predict 60 preious financial days will be used
#y_train contain the actual price
#each row in x_train correspond to the 60 previous timestamps for a given day
x_train = []
y_train = []
for i in range(80,1258):
    x_train.append(training_set[i-80:i,0:2])
    y_train.append(training_set[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)

regressor = Sequential()
regressor.add(LSTM(units = 125,return_sequences=True,
                   input_shape=(x_train.shape[1],2)))
#for training each data point there is 60 rows and 2 columns of data
#see x_train to understand how the data is prepared
#input shape- think of it as the data needed for one row/datapoint
#input_shape=(timesteps,features)
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 125,return_sequences=True))
regressor.add(LSTM(units = 125,return_sequences=True))
regressor.add(LSTM(units = 125, return_sequences = True))
regressor.add(LSTM(units = 125, return_sequences = True))
#regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 125,return_sequences=False))
#regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1, activation='linear'))
#rmsprop recommended for rnns according to keras documentation
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
regressor.fit(x_train,y_train,epochs=400,batch_size=16)

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
dataset_total = pd.concat((google_dataset[['Open','NASDAQ Open']],
                           test_set[['Open','NASDAQ Open']]),axis=0,
                          ignore_index=True)
inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values
scalar = StandardScaler()
inputs = scalar.fit_transform(inputs)

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
predicted_stock_price = scalar.inverse_transform(predicted_stock_price)
predicted_stock_price = predicted_stock_price.tolist()
for i in range(20):
    predicted_stock_price[i] = predicted_stock_price[i][0:1]
predicted_stock_price = np.array(predicted_stock_price)


hfm, = plt.plot(predicted_stock_price, 'b', label='predicted_stock_price')
hfm2 = plt.plot(actual_price_2017,'r', label = 'actual_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title("Actual vs predicted")
plt.title("Actual vs Predicted Stock price(Jan 2017)")
plt.ylabel("Price(USD)")
plt.xlabel("Time")
plt.savefig("Comparison.png")
plt.show()












    
