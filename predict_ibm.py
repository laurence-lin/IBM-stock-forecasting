import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data, wb
import pandas as pd
import datetime

sequence = 7
inputD = 1 # input dimension
outD = 1 # output dimension
#CODE BY SONG
def Normalized(data): # normalize data in each column

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / denominator

def dataConvert(data):
    data = data[::-1]
    data = Normalized(data)
    x_data = []
    y_data = []
    for i in range(0, len(data) - sequence):
        _x = data[i:i + sequence]
        _y = data[i + sequence]

        x_data.append(_x)
        y_data.append(_y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

#data = pd.read_csv('ibm.csv')
data = np.loadtxt('ibm2.csv')
a = []
for i in range(len(data)):
    a.append([data[i]])

data = np.array(a)
print(data.shape)
x_data, y_data = dataConvert(data)

train_size = int(len(y_data) * 0.7) # 70% data for training

x_train = np.array(x_data[:train_size])
x_test = np.array(x_data[train_size:])

y_train = np.array(y_data[:train_size])
y_test = np.array(y_data[train_size:])

print('tx:', x_train.shape,'tex:',x_test.shape,'ty:',y_train.shape,'tey:',y_test.shape)

X = tf.placeholder(tf.float32, [None, None, 1])
Y = tf.placeholder(tf.float32, [None, outD])

cell = rnn.BasicLSTMCell(num_units=10, state_is_tuple=True, activation=tf.tanh)

outputs, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], outD)

loss = tf.reduce_sum(tf.square(y_pred - Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

otherData = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
otherX, otherY = dataConvert(otherData)

x_train2 = x_train[:,:, np.newaxis]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1001):
        _, step_loss = sess.run([train, loss], feed_dict={X: x_train, Y: y_train})
        if(i % 100 == 0):
          print(i, step_loss)

    result = sess.run(y_pred, feed_dict={X:x_test})
    rmse = sess.run(rmse, feed_dict={targets:y_test, predictions:result})


print("rmse:",rmse)

plt.plot(y_test, 'red', label = 'real output')
plt.plot(result, 'b', label = 'predict output')
plt.legend(loc = 'upper right')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")

plt.figure(2)
plt.plot(data[7+train_size:])

plt.show()
