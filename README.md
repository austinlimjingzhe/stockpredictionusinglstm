# Stock Prediction Using LSTM

<h1>Introduction:</h1>

This repository serves to document what I learnt about stock price predictions using Long Short Term Memory (LSTM) networks.
This project references Aishwarya Singh’s blogpost on Analytics Vidhya. As Singh explains that stocks can be affected by company information (Singh, 2018), I attempt to integrate google trend data into the LSTM model.
The problem comes when extracting google trend data that is daily in nature when the range of dates required is more than 9 months. (Tseng, 2019) Fortunately, Tseng’s solution provides the solution that will be used for this repository.

<h1>Summarized Results:</h1>
My hypothesis for this project are that:

1. In the short term, the model that incorporates the google trend results would be more accurate as models using a longer time period could be using outdated sentiments.
2. Similar results when varying the lookback window.

Using Apple (AAPL) as an example, the root mean squared error (RMSE) are as follows:
| Length of Data Used (years) | W/O Google Trends Data |	With Google Trends Data  |
|-----------------------------|------------------------|--------------------------|
|             1               |	        15.44          |           6.70           |
|             2              	|          5.88	         |          12.11           |
|             3	              |          2.76          |          14.61           |

The results seem to be in line with the expectations.

<h1>Methodology</h1>
Firstly, import the packages and settings:

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
    os.chdir("Your Directory")

    import gtrend

    #to plot within notebook
    #%matplotlib inline
    rcParams['figure.figsize'] = 20,10
    
    #for normalizing data
    scaler = MinMaxScaler(feature_range=(0, 1))

<code>gtrend</code> is the python file that what written by Tseng to obtain the daily Google trends.

Next, download the relevant data using Yahoo Finance <code>yfinance</code> and Google trends <code>pytrends</code>
    
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
    
<h1>Exploratory Data Analysis</h1>

    #plot
    plt.figure(figsize=(16,8))
    plt.plot(df['Close'])
    plt.title("Close Price of "+ticker+" 2019-2021")
    plt.show()

    plt.figure(figsize=(16,8))
    plt.plot(data[keyword])
    plt.title("Google Search Trend of "+keyword)
    plt.show()

This code will print the time series graph of the price and trend.

<h1>Modelling</h1>
The next step would be to create the LSTM. We first prepare the data into a DataFrame.<br>
Using a 60 day or 2-month lookback, we split the data into train and test set.<br>
The LSTM is built using the keras framework and further studies can be conducted to find the best set of parameters for tuning.

    
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

    #predicting validation set
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

    #for plotting forecasts of validation set
    train = new_data[:int(0.8*n)]
    valid = new_data[int(0.8*n):]
    valid['Predictions'] = closing_price
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.title("Validation Set Predictions")
    plt.show()

The codes for the LSTM without incorporating google trends data can be found within this repository and are similar.

<h1>Forecasting:</h1>

Forecasting of future stock prices was attempted by making predictions 1 step at a time until the desired forecast window is reached for the model without incorporating google trends.<br><br>

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

The predictions would then be as follows:
![Predictions](https://user-images.githubusercontent.com/88301287/137916762-be50cada-f18d-49fe-b78b-c797e634a014.png)
 
One learning point along the way was the fact that in order to make future predictions, I would need to know not just the future values of the prices but also the future values of google trend results. One idea would be to have 2 other LSTM models to predict the future prices and future trend results to enter into the incorporated LSTM model 

<h1>Discussions</h1>
This beginner's attempt at creating LSTMs has much room to improve. Some suggestions include:

1. Varying the lookback window which is something that I was not able to get around to to confirm the second hypothesis.
2. Hyperparameter tuning for the LSTM
3. Implementing several LSTMs to allow for the forecasting of future prices as outlined in the Forecasting section.
4. Lastly, the ideal form of this project would be to package it in an application for users to be able to use. Quality-of-life improvements would thus have to include using a known list of companies and stock tickers as the current system requires users to know the ticker of the company they want to search.

<h1>References:</h1>

1.	Singh, A. (2021, July 23). Stock price prediction using machine learning: Deep learning. Analytics Vidhya. Retrieved October 19, 2021, from https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/. 
2.	Tseng, Q. (2019, November 29). Reconstruct google trends daily data for extended period. Medium. Retrieved October 19, 2021, from https://towardsdatascience.com/reconstruct-google-trends-daily-data-for-extended-period-75b6ca1d3420. 

 
