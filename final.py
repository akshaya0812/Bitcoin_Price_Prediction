import math
#import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.style.use ('fivethirtyeight')
import plotly.express as px
import datetime as dt
import yfinance as yf

import streamlit as st
from PIL import Image
import os
st.title('CRYPTO App')
st.write("Developed by : Akshaya and Keerthna")
#st.set_page_config(layout="wide") #st.beta_set_page_config(layout="wide")
image = Image.open(os.path.join('D:\\Akshaya\\crypto\\1.jpeg'))
st.image(image)

start = st.sidebar.date_input("Start date", dt.date(2021, 1, 1))
end = st.sidebar.date_input("End date", dt.date(2023, 1, 31))
#start = dt.date(2020,1,1)
#end = dt.date(2021,1,1)
#ticker = 'BTC'
com = st.text_input("Enter the Bitcoin Code of company",'BTC')
df = yf.download(com,start,end)
st.subheader('DATA INFORMATION')
st.dataframe(df)
df=df.dropna()

fig = go.Figure()
fig.add_trace(go.Scatter(x = df.index, y = df.High,mode='lines',name='High',marker_color = '#2CA02C',visible = "legendonly"))
fig.add_trace(go.Scatter(x = df.index, y = df.Low,mode='lines',name='Low',marker_color = '#D62728',visible = "legendonly"))
fig.add_trace(go.Scatter(x = df.index, y = df.Open,mode='lines',name='Open',marker_color = '#FF7F0E',visible = "legendonly"))
fig.add_trace(go.Scatter(x = df.index, y = df.Close,mode='lines',name='Close',marker_color = '#1F77B4'))

fig.update_layout(title='Closing price history',titlefont_size = 28,
                  xaxis = dict(title='Date',titlefont_size=16,tickfont_size=14),height = 800,
                  yaxis=dict(title='Price in INR (₹)',titlefont_size=16,tickfont_size=14),
                  legend=dict(y=0,x=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'))
fig.show()
st.write(fig)

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
training_data_len
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape
model = Sequential()  #initialize the RNN
model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))#adding input layerand the LSTM layer 
model.add(LSTM(50, return_sequences = False))#adding input layerand the LSTM layer 
model.add(Dense(25))
model.add(Dense(1)) #adding output layers
#model.compile(optimizer = 'adam', loss = 'mean_squared_error')  #compiling the RNN
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
history = model.fit(x_train, y_train, epochs=50, batch_size=16, shuffle=False ,validation_split=0.2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss graph')
plt.ylabel('Loss')
plt.xlabel('Epoch number')
plt.legend(loc="upper right")
plt.savefig("adam_loss_ethereum.png")
#plt.show()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
plt.plot(history.history['mean_squared_error'], label='Training MSE')
plt.plot(history.history['val_mean_squared_error'], label='Validation MSE')
plt.title('Mean Squared Error')
plt.ylabel('MSE value')
plt.xlabel('Epoch number')
plt.legend(loc="upper right")
plt.savefig("adam_mse_ethereum.png")
#plt.show()
st.pyplot()
#model.fit(x_train, y_train, batch_size = 1, epochs = 1) #fitting the RNN to the training set
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range (60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rsme = np.sqrt(np.mean(predictions - y_test) ** 2)
rsme
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
fig = go.Figure()
fig.add_trace(go.Scatter(x = train.index, y = train.Close,mode='lines',name='Close',marker_color = '#1F77B4'))
fig.add_trace(go.Scatter(x = valid.index, y = valid.Close,mode='lines',name='Val',marker_color = '#FF7F0E'))
fig.add_trace(go.Scatter(x = valid.index, y = valid.Predictions,mode='lines',name='Predictions',marker_color = '#2CA02C'))

fig.update_layout(title='Model',titlefont_size = 28,hovermode = 'x',
                  xaxis = dict(title='Date',titlefont_size=16,tickfont_size=14),height = 800,
                  yaxis=dict(title='Close price in INR (₹)',titlefont_size=16,tickfont_size=14),
                  legend=dict(y=0,x=1.0,bgcolor='rgba(255, 255, 255, 0)',bordercolor='rgba(255, 255, 255, 0)'))
fig.show()
st.write(fig)
