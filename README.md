# GoogleStock
Build a Recurrent Neural Network to analyze the Google Stock (NASDAQ:GOOGL).
> >-Training set contains information from 2012 to 2016.\
> >-Test Set contains information from the first few days from 2017.
### Data Preprocessing
1. The two features considered are-
> >* Opening price of the NASDAQ index. The rational behind this is that the google stock is one of the top companies that forms this index 
and the price movement between this index and the stock are very similar. 
> >* Opening price of the Google stock

2. CSV files for the NASDAQ index and the google stock were obtained from Yahoo Finance. 
3. In order to train the model, training data consists of 60 previous opening prices of the google stock for a single data point along with the corresponding opening price of the NASDAQ.
4. Important note- KEEP THE TRAINING AND TEST DATA IN THE FORM OF NUMPY ARRAYS.

### About the model
1. 4 LSTM layers were considered. 
2. Dropout regularization was also added in order to prevent overfitting of the training data. 
3. The model fit the data over 100 epochs with a batch size of 32. 



