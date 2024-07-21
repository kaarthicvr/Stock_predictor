import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the model
try:
    model = tf.keras.models.load_model('Stock prediction model.keras')
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

def fetch_stock_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        data['MA50'] = data['Close'].rolling(50).mean()
        data['MA100'] = data['Close'].rolling(100).mean()
        data['MA200'] = data['Close'].rolling(200).mean()
        print("Columns in data:", data.columns)  # Debugging print statement
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}", file=sys.stderr)
        raise

def predict(symbol):
    try:
        # Fetch stock data
        data = fetch_stock_data(symbol, '2018-01-01', '2023-12-31')
        
        # Check if columns exist
        if not all(col in data.columns for col in ['MA50', 'MA100', 'MA200']):
            raise ValueError("One or more required columns are missing from the data")
        
        # Prepare data for prediction
        data = data[['Close']]
        data = data.dropna()

        data_train = pd.DataFrame(data[:int(len(data) * 0.80)])
        past_100_days = data_train.tail(100).values

        scaler = MinMaxScaler(feature_range=(0, 1))
        past_100_days_scaled = scaler.fit_transform(past_100_days)

        x_input = past_100_days_scaled.reshape((1, past_100_days_scaled.shape[0], 1))

        # Make prediction
        y_predict = model.predict(x_input)

        # Inverse scaling
        scale = 1 / scaler.scale_[0]
        y_predict = y_predict * scale

        data_test = pd.DataFrame(data[int(len(data) * 0.80):])
        data_test = pd.concat([data_train.tail(100), data_test], ignore_index=True)
        data_test_scaled = scaler.fit_transform(data_test)

        x_test = []
        y_test = []

        for i in range(100, data_test_scaled.shape[0]):
            x_test.append(data_test_scaled[i-100:i])
            y_test.append(data_test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predict = model.predict(x_test)
        y_predict = y_predict * scale
        y_test = y_test * scale

        return {
            'ma50': data['MA50'].dropna().tolist(),
            'ma100': data['MA100'].dropna().tolist(),
            'ma200': data['MA200'].dropna().tolist(),
            'original_prices': y_test.tolist(),
            'predicted_prices': y_predict.tolist(),
            'close_prices': data['Close'].dropna().tolist()
        }
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    print("Arguments:", sys.argv)  # Debugging print statement
    if len(sys.argv) > 1:
        try:
            data = json.loads(sys.argv[1])
            symbol = data['symbol']
            prediction = predict(symbol)
            print(json.dumps(prediction))
        except Exception as e:
            print(f"Error in main execution: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No arguments provided.")
