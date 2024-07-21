import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to bottom right, blue, white);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: black;'>Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: orange;'>Project done by Kaarthic VR</h2>", unsafe_allow_html=True)

model = tf.keras.models.load_model('Stock prediction model.keras')

stock = st.text_input('Enter Stock Symbol', 'GOOG')

start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)
st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')

ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(5, 4))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')

ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(5, 4))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')

ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(5, 4))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')

fig4 = plt.figure(figsize=(5, 4))
plt.plot(range(len(y)), y, 'g', label='Original Price')
plt.plot(range(len(predict)), predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
