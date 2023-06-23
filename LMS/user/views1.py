from django.shortcuts import render
import pandas as pd
import yfinance as yf 
import plotly.offline as pyo
import plotly.graph_objs as go
from datetime import datetime,timedelta
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split



#     all=pd.DataFrame(yf.download(tickers=['TSLA', "HMC", "F"],start='2020-01-01',end='2022-01-07',progress=False,))
# #    print(all)
#     trace = go.Scatter(
#     x=all.index,
#     y=all["Close"]["F"],
#     mode='lines',
#     name='Ford Stock'
#     )
#     trace1 = go.Scatter(
#     x=all.index,
#     y=all['Close']['HMC'],
#     mode='lines',
#     name='Honda Stock'
#     )
#     trace2 = go.Scatter(
#     x=all.index,
#     y=all['Close']['TSLA'],
#     mode='lines',
#     name='Tesla stock'
#     )
#     data = [trace, trace1,trace2]
#     layout = go.Layout(
#     title='Time Series Data',
#     xaxis=dict(title='Timestamp'),
#     yaxis=dict(title='Value')
#     )
#     fig = go.Figure(data=data, layout=layout)
#     plot_div = pyo.plot(fig, output_type='div', config= {'displaylogo': False}, include_plotlyjs=False)

#     # history = yf.download("TSLA", start="2022-11-01", end="2022-12-31")

#     # dates = history.index.strftime("%Y-%m-%d")
#     # prices_tesla = history["Close"].tolist()
#     # context = {
#     #     "dates": dates,
#     #     "prices_tesla": prices_tesla
#     #  }
#     return render(request,'user/index.html',{'div':plot_div})    

    # Get the stock data from Yahoo Finance
def index(request):
    stocks = ["TSLA","HMC"]  #, "HMC", "F"
    data={}
    
    for stock in stocks:
        data[stock]=datagenerate(stock)
        print(data[stock].iloc[:-5])
    
    return render(request,'user/index.html',{'div':0,'context':'p'})

def datagenerate(stocks):    
    
    history = yf.download(stocks, start="2022-01-01", end="2022-12-31")
    # Extract the date and closing prices for each stock
#    global df
    df= pd.DataFrame(history)
    # Convert the 'Date' column to datetime format
    # df['Date'] = pd.to_datetime(df['Date'])

    # # Set the 'Date' column as the index
    # df.set_index('Date', inplace=True)

    return df


def preprocessing(stock_data):
    # Create a dataframe from the stock data

    # Select the 'Close' price as the target variable
    target_variable = 'Close'

    # Select the features and the target variable
    features = ['Open', 'High', 'Low', 'Volume']
    target = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[features], target, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def train(X_train, X_test, y_train, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model
def predict(model):
    next_week_dates = pd.date_range(end=df.index[-1], periods=7, freq='D')
    predictions_df = pd.DataFrame(index=next_week_dates, columns=['Predicted Price'])
    for date in next_week_dates:
        prediction = model.predict(last_row)
        predictions_df.loc[date, 'Predicted Price'] = prediction[0]
        last_row = last_row.shift(-1)
        last_row.iloc[-1] = prediction[0]
    return predictions_df
    





    
    # Perform stock prediction for one more week
    future_dates = [datetime.strptime(dates[-1], "%Y-%m-%d") + timedelta(days=i) for i in range(1, 8)]
    future_dates = [date.strftime("%Y-%m-%d") for date in future_dates]


    model = xgb.XGBRegressor()
    model.fit(df)

    # Make future predictions
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    predicted_prices = forecast["yhat"].tail(7).tolist()

    # Pass the data to the template
    context = {
        "dates": dates,
        "prices_tesla": prices_tesla,
        "prices_honda": prices_honda,
        "prices_ford": prices_ford,
        "future_dates": future_dates,
        "predicted_prices": predicted_prices,
    }
    return render(request, "user/index.html", context)
