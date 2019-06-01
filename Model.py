import tensorflow as tf


def create_model(stride, length=20, input=4):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(length, input_shape=(stride - 1, input), return_sequences=True))
    model.add(tf.keras.layers.LSTM(length))
    model.add(tf.keras.layers.Dense(input, activation=tf.nn.relu))

    return model
