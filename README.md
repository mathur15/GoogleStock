# GoogleStock
Build a Recurrent Neural Network to analyze the Google Stock (NASDAQ:GOOGL). The goal is to predict the google stock prices in the first financial month of 2017 using a model which will be trained on data from the past 5 years. 
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
5. Data was standardized using StandardScaler. Different StandardScaler objects were used on the two indicators as pre caution as both indicator consisted of data on different scales. 
6. Data was cleaned to remove "," from the data. 

### About the model
1. 4 LSTM layers were considered. 
2. Dropout regularization was also added in order to prevent overfitting of the training data. 
3. The model fit the data over 100 epochs with a batch size of 32 initially. The last value tried out was 400 epochs with a batch size of 16. 

### Challenged going forward
1. While the model was able to predict smooth changes fairly accurately based on the visualization created, there was some discrpency with regards to time periods when the stock showed a lot of volatility. 
2. While it is difficult to predict future variation in pricing of a stock on past trends here are some methods I will use to further optimize the  model- 
   * Experiment with the complexity of the LSTM layer by adding more layers or increasing the number of units for each layer.
   * Train the model on more past timestamps, instead of 60(current value chosen) previous timestamps perhaps try higher values in the range of 120-150. 
   * Incorporate more indicators related to the stock like it's closing price, volume, etc. 
   * Train the model on more past data. Currently the model is being trained on data from 2012-2016 to predict the price points for first financial month of 2017. Going forward, the goal will be to use the past 10 years data of the stock to develop a more robust model. 



