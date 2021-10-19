# stockpredictionusinglstm
Introduction:

This repository serves to document what I learnt about stock price predictions using Long Short Term Memory (LSTM) networks.
This project references Aishwarya Singh’s blogpost on Analytics Vidhya. As Singh explains that stocks can be affected by company information (Singh, 2018), I attempt to integrate google trend data into the LSTM model.
The problem comes when extracting google trend data that is daily in nature when the range of dates required is more than 9 months. (Tseng, 2019) Fortunately, Tseng’s solution provides the solution that will be used for this repository.

Results:

Using Apple (AAPL) as an example, the root mean squared error (RMSE) are as follows:
| Length of Data Used (years) | W/O Google Trends Data |	With Google Trends Data |
|-----------------------------|------------------------|--------------------------|
|             1               |	        15.44          |           6.70           |
|             2             	|          5.88	         |          12.11           |
|             3	              |          2.76          |          14.61           |

Forecasting:

Forecasting of future stock prices was attempted by making predictions 1 step at a time until the desired forecast window is reached.
![Predictions](https://user-images.githubusercontent.com/88301287/137916762-be50cada-f18d-49fe-b78b-c797e634a014.png)
 
References:

1.	Singh, A. (2021, July 23). Stock price prediction using machine learning: Deep learning. Analytics Vidhya. Retrieved October 19, 2021, from https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/. 
2.	Tseng, Q. (2019, November 29). Reconstruct google trends daily data for extended period. Medium. Retrieved October 19, 2021, from https://towardsdatascience.com/reconstruct-google-trends-daily-data-for-extended-period-75b6ca1d3420. 

 
