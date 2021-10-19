#import packages
import os
import pandas as pd
import numpy as np
import dateutil.relativedelta
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from pytrends.request import TrendReq

os.getcwd()
os.chdir("C:\\Users\\User\\OneDrive\\Desktop\\Stock Price Forecasting")
os.getcwd()

import gtrend

#to plot within notebook
#%matplotlib inline
rcParams['figure.figsize'] = 20,10

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

#download stock
lookback=int(input("How many years worth of data do you want to train on? (Please input an integer) "))
start_date=str(date.today()-dateutil.relativedelta.relativedelta(years=lookback))
end_date=str(date.today())
ticker = input("What stock ticker do you want to view? ")
df = yf.download(ticker,start_date,end_date)
df.head()

#get google trends data
pytrends=TrendReq(hl="en-US",tz=360)
keyword=input("Please enter the corresponding search term (company name): ")
results=gtrend.get_daily_trend(pytrends, keyword, start_date, end_date, geo="", cat=0, gprop="", verbose=True)
data=df.merge(results,left_index=True,right_index=True)

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.title("Close Price of "+ticker+" 2019-2021")
plt.show()

plt.figure(figsize=(16,8))
plt.plot(data[keyword])
plt.title("Google Search Trend of "+keyword)
plt.show()

#data preparation
new_data = pd.DataFrame(data[['Close',keyword]])
dataset = new_data.values

n=len(dataset)
train = dataset[0:int(0.8*n),]
valid = dataset[int(0.8*n):,:]

scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,:])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],2))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=500, return_sequences=True, input_shape=(x_train.shape[1],2)))
model.add(Dropout(0.2))
model.add(LSTM(units=500))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=60, verbose=2,callbacks=[es])

#predicting new forecasts
inputs = new_data[len(new_data) - len(valid) - 60:].values

inputs = inputs.reshape(-1,2)
inputs  = scaler.transform(inputs)

X_test = []
for j in range(60,inputs.shape[0]):
    X_test.append(inputs[j-60:j,:])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],2))
closing_price = model.predict(X_test)
extra_col = np.zeros((closing_price.shape[0],1))
closing_price = np.append(closing_price,extra_col,axis=1)
closing_price = scaler.inverse_transform(closing_price)
closing_price = np.delete(closing_price,1,axis=1)

rms=np.sqrt(np.mean(np.power((valid[:,0]-closing_price),2)))
rms

#for plotting forecasts
train = new_data[:int(0.8*n)]
valid = new_data[int(0.8*n):]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.title("Validation Set Predictions")
plt.show()