import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
from DataManagement import *
from OnlineStockData import *
from Model import *

EPOCHS = 10
STRIDE = 40
raw_path = 'amd.us.txt'
chosen_stock = 'amd'
start = datetime.datetime(1984, 1, 1)
end = datetime.datetime.now()

#raw_data = pd.read_csv(raw_path, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
raw_data, ticker = get_stock_data(start_date=start, end_date=end)
raw_data.to_csv("stock_prices.csv")
raw_data = pd.read_csv("stock_prices.csv", delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
print(raw_data.loc[0, :])  # first row and all columns

plt.plot(range(raw_data.shape[0]), (raw_data['Low'] + raw_data['High']) / 2.0)
plt.xticks(range(0, raw_data.shape[0], 500), raw_data['Date'].loc[::500], rotation=45)
plt.title('{} Stock Price History'.format(ticker))
plt.xlabel('Date')
plt.ylabel('Average Day Price')
plt.show()


# Normalize data and create training sets
avg_cost = raw_data.mean(axis = 1)
scaler = MinMaxScaler(feature_range=(0,1))
avg_cost = scaler.fit_transform(np.reshape(avg_cost.values, (len(avg_cost), 1)))
norm_data = normalize(raw_data, ['Open', 'High', 'Low', 'Close'], 0, 1)
norm_data = norm_data[['Open', 'High', 'Low', 'Close'].copy()]
x_train, y_train, x_test, y_test = train_test_sets(norm_data, STRIDE)


# Create the LSTM model
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.LSTM(20, input_shape=(STRIDE-1, 4), return_sequences=True))
#model.add(tf.keras.layers.LSTM(20))
#model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))
model = create_model(stride=STRIDE)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])


# Run the model on the training data and gather statistics
model.fit(x_train, y_train, epochs=EPOCHS)
test_loss, test_acc = model.evaluate(x_train, y_train)
print('Test accuracy:', test_acc)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)     # un-normalize data to get representative price


# Plot the test predictions vs true prices
predictions = predictions[:,[1,2]]  # get only the high and low
predictions_avg = np.mean(predictions, axis=1)

true_price = ((raw_data['Low'] + raw_data['High']) / 2.0)[len(x_train):]
true_price = np.array(true_price)[STRIDE:]
#padded_pred = np.pad(predictions, (len(x_train) + len(x_valid), 0), 'constant')
plt.clf()
#plt.xticks(range(0, raw_data.shape[0], 500), raw_data['Date'].loc[::500], rotation=45)
plt.plot(true_price, label='True Price')
plt.plot(predictions_avg, label="Test Estimated")
plt.legend()
plt.title('{} Price Prediction on New Data'.format(ticker))
plt.xlabel('Days')
plt.ylabel('Average Day Price')
plt.show()


# Plot the training predictions vs true prices
predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)

predictions = predictions[:,[1,2]]  # get only the high and low
predictions_avg = np.mean(predictions, axis=1)
true_price = ((raw_data['Low'] + raw_data['High']) / 2.0)[:len(predictions_avg)]
plt.clf()
plt.plot(true_price, label='True Price')
plt.plot(predictions_avg, label="Training Estimated")
plt.legend()
plt.title('{} Price Prediction During Training'.format(ticker))
plt.xlabel('Days')
plt.ylabel('Average Day Price')
plt.show()

