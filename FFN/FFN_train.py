import tensorflow as tf
import pandas as pd
from tkinter import Tk, filedialog
import numpy as np
import os
import shutil


# create folder for summaries (tensorboard)
if not os.path.exists('summaries'):
    os.mkdir('summaries')
if os.path.exists(os.path.join('summaries','first')):
    shutil.rmtree('summaries')
    os.mkdir('summaries')

os.mkdir(os.path.join('summaries','first'))

# neural network parameters
learning_rate = 0.001
training_epochs = 100
train_test_ratio = 0.7
num_out = 1
display_step = 5
layer_1 = 50     # hidden layers
layer_2 = 50
layer_3 = 50

# loading input file
Tk().withdraw()
fname = filedialog.askopenfile(title="Select the training file")
data = pd.read_csv(fname, dtype=float)
msk = np.random.rand(data.shape[0]) < train_test_ratio
training_data = data[msk]
testing_data = data[~msk]

# get the training columns
num_in = training_data.shape[1]-num_out
X = training_data.ix[:, range(0, num_in)].values
Y = training_data.ix[:, num_in].values

# get the testing columns
Xt = testing_data.ix[:, range(0, num_in)].values
Yt = testing_data.ix[:, num_in].values

# normalize the data
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_sc = (X - X_min) / (X_max - X_min)
Y_min = Y.min(axis=0)
Y_max = Y.max(axis=0)
Y_sc = (Y - Y_min) / (Y_max - Y_min)
Y_sc = Y_sc.reshape(Y_sc.size, 1)

# fit testing data
Xt_sc = (Xt - X_min) / (X_max - X_min)
Yt_sc = (Yt - Y_min) / (Y_max - Y_min)
Yt_sc = Yt_sc.reshape(Yt_sc.size, 1)

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
cost_summ = tf.summary.scalar('loss function value', cost)

# Optimizers
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Savers
saver = tf.train.Saver()

# initialize tensorflow session
with tf.Session() as session:

    # summary writers
    # write the graph to the summary
    merged = tf.summary.merge_all()
    summ_writer = tf.summary.FileWriter(os.path.join('summaries','first'), session.graph)

    session.run(tf.global_variables_initializer())

    # run the training loop
    for epoch in range(training_epochs):
        summary, acc = session.run([cost_summ, optimizer], feed_dict={X: X_sc, Y: Y_sc})

        # add accuracy to the summaries
        summ_writer.add_summary(summary, epoch)
        
        if epoch % display_step == 0:
            training_cost = session.run(cost, feed_dict={X: X_sc, Y: Y_sc})
            testing_cost = session.run(cost, feed_dict={X: Xt_sc, Y: Yt_sc})
            print("Epoch:", epoch, "Training:", training_cost, "Testing:", testing_cost)


    # store the scaler and trained model parameters
    X_min = np.reshape((np.array(X_min)), (-1, 1))
    X_max = np.reshape((np.array(X_max)), (-1, 1))
    Y_min = np.reshape((np.array(Y_min)), (-1, 1))
    Y_max = np.reshape((np.array(Y_max)), (-1, 1))
    num_in = np.reshape((np.array(num_in)), (-1, 1))
    num_out = np.reshape((np.array(num_out)), (-1, 1))
    min_params = np.concatenate((X_min, Y_min, num_in), axis=0)
    max_params = np.concatenate((X_max, Y_max, num_out), axis=0)
    scaler_params = pd.DataFrame(np.concatenate((min_params, max_params), axis=1))
    scaler_params.to_csv("models/scaler.csv")

    save_path = saver.save(session, "models/trained.ckpt")
    print("Model saved: {}".format(save_path))

