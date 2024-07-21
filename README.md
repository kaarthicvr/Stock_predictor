# Stock Market Predictor

## Project Description

The Stock Market Predictor is a web application built using Streamlit. It allows users to predict stock prices using a machine learning model. The project uses historical stock data to train a model and make future predictions. The application also visualizes the stock data and various moving averages.

## Project done by Kaarthic VR

## Features

- Download historical stock data from Yahoo Finance.
- Visualize stock prices along with moving averages (MA50, MA100, MA200).
- Predict future stock prices using a pre-trained machine learning model.

## Installation

1. Clone the repository:

    ghttps://github.com/kaarthicvr/Stock_predictor.git

2. Create and activate a virtual environment (optional but recommended):
    
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required libraries:

    pip install -r requirements.txt
  
## Usage

1. Run the Streamlit app:

    streamlit run app.py

2. Open your web browser and go to `http://localhost:8501`.

3. Enter a stock symbol (e.g., `GOOG`) and view the historical stock data along with various moving averages.

4. The application will display the original stock prices vs. the predicted stock prices.

## Project Structure

- `app.py`: Main Streamlit application script.
- `requirements.txt`: List of required Python libraries.

## Libraries Used

- numpy
- pandas
- yfinance
- tensorflow
- keras
- streamlit
- matplotlib
- scikit-learn

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

