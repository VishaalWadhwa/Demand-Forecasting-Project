from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def train_arima(train_data):
    arima_model = ARIMA(train_data['sales'], order=(5, 1, 0))
    arima_fit = arima_model.fit()
    return arima_fit

def train_random_forest(train_data, features):
    rf_model = RandomForestRegressor()
    rf_model.fit(train_data[features], train_data['sales'])
    return rf_model

def train_lstm(train_data, features):
    train_lstm = np.array(train_data[features]).reshape((train_data.shape[0], 1, len(features)))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(1, len(features))))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    
    lstm_model.fit(train_lstm, train_data['sales'], epochs=50, verbose=0)
    return lstm_model
