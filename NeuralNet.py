import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
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


norm_data = normalize(raw_data, ['Open', 'High', 'Low', 'Close'])
norm_data = norm_data[['Open', 'High', 'Low', 'Close'].copy()]
x_train, y_train, x_valid, y_valid, x_test, y_test = create_training_sets(norm_data, 40)
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(20, input_shape=(39, 4), return_sequences=True))
model.add(tf.keras.layers.LSTM(20))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mean_squared_error")

plt.clf()
plt.plot(y_train)
#plt.show()
model.fit(x_train, y_train, epochs=1)
test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(x_test)
print(predictions[0])

