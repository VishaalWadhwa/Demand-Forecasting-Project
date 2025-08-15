from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def evaluate_model(test_data, predictions, model_name):
    mae = mean_absolute_error(test_data['sales'], predictions)
    mse = mean_squared_error(test_data['sales'], predictions)
    mape = mean_absolute_percentage_error(test_data['sales'], predictions)
    print(f'{model_name} MAE: {mae}')
    print(f'{model_name} MSE: {mse}')
    print(f'{model_name} MAPE: {mape}')
    return mae, mse, mape

def plot_forecasts(test_data, arima_forecast, rf_forecast, lstm_forecast):
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data['sales'], label='Actual Sales')
    plt.plot(test_data.index, arima_forecast, label='ARIMA Forecast')
    plt.plot(test_data.index, rf_forecast, label='Random Forest Forecast')
    plt.plot(test_data.index, lstm_forecast, label='LSTM Forecast')
    plt.legend()
    plt.show()
