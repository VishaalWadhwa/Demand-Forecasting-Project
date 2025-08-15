import pandas as pd

def load_and_preprocess_data(sales_file, trends_file):
    sales_data = pd.read_csv(sales_file)
    market_trends = pd.read_csv(trends_file)
    
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    sales_data.set_index('date', inplace=True)
    
    sales_data['lag_1'] = sales_data['sales'].shift(1)
    sales_data['rolling_mean_7'] = sales_data['sales'].rolling(window=7).mean()
    
    sales_data.dropna(inplace=True)
    
    return sales_data, market_trends
