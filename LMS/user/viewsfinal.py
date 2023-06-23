
from django.shortcuts import render
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional

def build_lstm_model(num_layers, num_units,time_steps,num_features, bidirectional=False):
    model = Sequential()

    if bidirectional:
        # Add a bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=num_units, return_sequences=True), input_shape=(time_steps, num_features)))
    else:
        # Add the first LSTM layer
        model.add(LSTM(units=num_units, return_sequences=True, input_shape=(time_steps, num_features)))

    # Add additional LSTM layers
    for _ in range(2, num_layers):
        model.add(LSTM(units=num_units, return_sequences=True))

    model.add(LSTM(units=num_units))
    # Add a final dense layer
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')


    return model

def calculate_technical_indicators(df, window_size=14):
    # Calculate Moving Averages (MA)
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_30'] = df['Close'].rolling(window=30).mean()

    # Calculate Relative Strength Index (RSI)
    delta = df['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=window_size).mean()
    avg_loss = abs(down.rolling(window=window_size).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    df['month']=df.index.month
    df['day']=df.index.day_of_week
    df['week']=df.index.weekofyear

    return df

def index(request):
    ticker = input("Stock name")
    df = yf.download(ticker)#,start=start_date) #, start='max', end=(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    df=calculate_technical_indicators(df)
    df.dropna(axis=0,inplace=True)
    columns = df.columns.to_list()
    data = df[columns].values

    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create training data with sequences of the last 30 days' data
    X_train = []
    y_train = []
    window_size = 30

    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i - window_size:i])
        y_train.append(scaled_data[i][3])  # Predicting the 'Close' column value

    # Convert the training data to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Reshape the input data for the LSTM model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(columns)))
    model=build_lstm_model(3,50,X_train.shape[1],len(columns))

    last_30_days = scaled_data[-window_size:]

    # Reshape the input data for prediction
    X_pred = np.array([last_30_days])
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], len(columns)))
    predicted_prices = []

    for _ in range(7):
        # Predict the next day's stock price
        prediction = model.predict(X_pred)
        
        # Denormalize the predicted value
        predicted_close = scaler.inverse_transform([[0,0,0, prediction[0][0],0, 0,0,0,0,0, 0, 0]])[0][3]

        # Append the predicted price to the list
        predicted_prices.append(predicted_close)

        # Update the input data for the next prediction
        X_pred = np.roll(X_pred, -1, axis=1)
        X_pred[0, -1] = prediction[0]
        
    
    pre=pd.DataFrame(index=pd.date_range(start=df.index[-1]+pd.DateOffset(days=1), periods=7,freq='B'),data=predicted_prices,columns=['Close'])
    tesl=pd.concat([df[['Close']],pre])
    
    div=plot(tesl)
    
    return render(request,'user/index.html')

    


        
