#import packages
import pandas as pd
import numpy as np
import dateutil.relativedelta
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
import datetime
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM

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

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'])
plt.title("Close Price of "+ticker)
plt.show()

#data preparation
new_data = pd.DataFrame(df['Close'])
dataset = new_data.values

n=len(dataset)
train = dataset[0:int(0.8*n),]
valid = dataset[int(0.8*n):,:]

scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=500, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=500))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
es = EarlyStopping(monitor='loss', mode='min', verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=60, verbose=2,callbacks=[es])

#predicting new forecasts
inputs = new_data[len(new_data) - len(valid) - 60:].values

inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs) 

X_test = []
for j in range(60,inputs.shape[0]):
    X_test.append(inputs[j-60:j,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms

#for plotting forecasts
train = new_data[:int(0.8*n)]
valid = new_data[int(0.8*n):]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.title("Validation Set Predictions")
plt.show()

#forecast forwards
forecast_length=int(input("How many days do you want to forecast? "))
new_dates=pd.date_range(start=max(new_data.index)+datetime.timedelta(days=1), periods=forecast_length, freq='B')

newinputs = new_data[len(new_data) - 60 :].values
newinputs = newinputs.reshape(-1,1)
newinputs  = scaler.transform(newinputs) 

predictions=np.empty((0,1))

while forecast_length>0:
    forecast=[newinputs[:,0]]
    forecast=np.array(forecast)
    forecast=np.reshape(forecast,(forecast.shape[0],forecast.shape[1],1))
    newprice=model.predict(forecast)
    
    predictions=np.append(predictions,newprice.item())
    newinputs=np.append(newinputs,newprice.item())
    newinputs=np.delete(newinputs, 0)
    newinputs=np.reshape(newinputs,(newinputs.shape[0],1))
    forecast_length-=1

#plotting the forecast predictions
predictions = np.reshape(predictions,(predictions.shape[0],1))    
predictions = scaler.inverse_transform(predictions)
future=pd.DataFrame(predictions,columns=["Prediction"],index=new_dates)
ax=future.plot(ax=new_data.plot(title="Stock Price of "+ticker)).axvline(max(new_data.index),color="r",linestyle="--")

