import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from DataManagement import *


raw_path = os.path.join('../Bigzip/Stocks', 'amd.us.txt')
raw_data = pd.read_csv(raw_path, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
print(raw_data.loc[0, :])  # first row and all columns

plt.title('AMD Stock Price History')
plt.plot(range(raw_data.shape[0]), (raw_data['Low'] + raw_data['High']) / 2.0)
plt.xticks(range(0, raw_data.shape[0], 500), raw_data['Date'].loc[::500], rotation=45)
plt.xlabel('Date')
plt.ylabel('Average Day Price')
#plt.show()

avg_cost = raw_data.mean(axis = 1)
scaler = MinMaxScaler(feature_range=(0,1))
avg_cost = scaler.fit_transform(np.reshape(avg_cost.values, (len(avg_cost), 1)))
norm_data = normalize(raw_data, ['Open', 'High', 'Low', 'Close'], 0, 1)
norm_data = norm_data[['Open', 'High', 'Low', 'Close'].copy()]
x_train, y_train, x_valid, y_valid, x_test, y_test = create_training_sets(norm_data, 40)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(39, 4), return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(x_train, y_train, epochs=10)
test_acc = model.evaluate(x_train, y_train)
print('Test accuracy:', test_acc)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print(predictions)

predictions = predictions[:,[1,2]]  # get only the high and low
predictions_avg = np.mean(predictions, axis=1)
true_price = ((raw_data['Low'] + raw_data['High']) / 2.0)[:len(predictions_avg)]
#padded_pred = np.pad(predictions, (len(x_train) + len(x_valid), 0), 'constant')
plt.clf()
#plt.xticks(range(0, raw_data.shape[0], 500), raw_data['Date'].loc[::500], rotation=45)
plt.plot(true_price, label='True Price')
plt.plot(predictions_avg, label="Test Estimated")
plt.legend()
plt.show()


predictions = model.predict(x_train)
predictions = scaler.inverse_transform(predictions)
print(predictions)

predictions = predictions[:,[1,2]]  # get only the high and low
predictions_avg = np.mean(predictions, axis=1)
true_price = ((raw_data['Low'] + raw_data['High']) / 2.0)[:len(predictions_avg)]
plt.clf()
plt.plot(true_price, label='True Price')
plt.plot(predictions_avg, label="Training Estimated")
plt.legend()
plt.show()

