# TensorFlow Samples
Simplified and easy to follow Neural networks built with Tensorflow

## 1. Feed Forward Neural Network
   This is a multi-layer neural network in which the data flows in one direction from the input layer to the output
layer. The overview of the network is shown below. There is an input layer and three hidden layers and an output layer

   ![alt text](https://raw.githubusercontent.com/ashyantony7/TensorFlow_Samples/master/doc/FFN_1.png)

### Instructions
#### (i) Training
  1. Run FFN_Train.py
  2. Select the data file 
  3. To view the graphs and loss functions enter in Terminal 
  `  tensorboard --logdir=summaries`
       or
  `  python -m tensorboard.main --logdir=summaries`
  
  The loss functions can be viewed as a scalar in Tensorboard below

  ![alt text](https://raw.githubusercontent.com/ashyantony7/TensorFlow_Samples/master/doc/FFN_2.png)
