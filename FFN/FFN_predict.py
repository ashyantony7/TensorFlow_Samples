import tensorflow as tf
import pandas as pd
from tkinter import Tk, filedialog
import os

# load the model folder
Tk().withdraw()
dirname = filedialog.askdirectory(title="select the trained model")
filepath = os.path.join(dirname, "scaler.csv")
scaler_params = pd.read_csv(filepath, dtype=float)
max_params = scaler_params.ix[:, -1]
min_params = scaler_params.ix[:, -2]
num_in = int(min_params[len(min_params)-1])
num_out = int(max_params[len(max_params)-1])

# loading CSV file
# Tk().withdraw()
fname = filedialog.askopenfile(title="Select the testing file")

# get the testing columns
testing_data = pd.read_csv(fname, dtype=float)
Xt = testing_data.ix[:, range(0, num_in)].values
Yt = testing_data.ix[:, num_in].values

# fit testing data
X_min = min_params[list(range(0, num_in))].values
X_max = max_params[list(range(0, num_in))].values
Y_min = min_params[list(range(num_in, (num_in + num_out)))].values
Y_max = max_params[list(range(num_in, (num_in + num_out)))].values
Xt_sc = (Xt - X_min) / (X_max - X_min)
Yt_sc = (Yt - Y_min) / (Y_max - Y_min)
Yt_sc = Yt_sc.reshape(Yt_sc.size, 1)

# hidden layers
layer_1 = 50
layer_2 = 50
layer_3 = 50

# detail definition of network

# Input layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, num_in))

# Hidden layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights1", shape=[num_in, layer_1],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1], initializer=tf.zeros_initializer())
    layer_1_out = tf.nn.relu(tf.matmul(X, weights) + biases)

# Hidden layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1, layer_2],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2], initializer=tf.zeros_initializer())
    layer_2_out = tf.nn.relu(tf.matmul(layer_1_out, weights) + biases)

# Hidden layer 3
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2, layer_3],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3], initializer=tf.zeros_initializer())
    layer_3_out = tf.nn.relu(tf.matmul(layer_2_out, weights) + biases)

# Output layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3, num_out],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[num_out], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_out, weights) + biases)

# Cost function
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Savers
saver = tf.train.Saver()

# initialize tensorflow session
with tf.Session() as session:

    saver.restore(session, "models/trained.ckpt")
    # run the training loop
    testing_cost = session.run(cost, feed_dict={X: Xt_sc, Y: Yt_sc})
    print("Mean Error", testing_cost)