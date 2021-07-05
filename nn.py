
#-------------------------------------------------------------------------------
# nn.py written by Nelson Rayl 2021-06-26 
# 
# Creates neural network from scratch in numpy.
#-------------------------------------------------------------------------------

import os
import gzip
import math
import numpy as np

os.chdir('/Users/nelsonrayl/Desktop/init/fp_NN')

data_dir = '/Users/nelsonrayl/Desktop/init/fp_NN/data/'

#===============================================================================
# Unzip Data and Convert to .np
#===============================================================================

# Unzip and read MNIST files.
train_imgs = gzip.open(data_dir + 'train-images-idx3-ubyte.gz', 'rb').read()
train_lbls = gzip.open(data_dir + 'train-labels-idx1-ubyte.gz', 'rb').read()
test_imgs  = gzip.open(data_dir + 't10k-images-idx3-ubyte.gz', 'rb').read()
test_lbls  = gzip.open(data_dir + 't10k-labels-idx1-ubyte.gz', 'rb').read()

# Convert to ndarray, reshape if necessary.
train_imgs = np.frombuffer(train_imgs, dtype=np.uint8, offset=16).reshape(-1,28*28)/255
train_lbls = np.frombuffer(train_lbls, dtype=np.uint8, offset=8)
test_imgs  = np.frombuffer(test_imgs, dtype=np.uint8, offset=16).reshape(-1,28*28)/255
test_lbls  = np.frombuffer(test_lbls, dtype=np.uint8, offset=8)

#===============================================================================
# Define Activation Functions and Derivatives of Activation Functions
#===============================================================================

def relu(x): 
    x[x<0]=0
    return x

def relu_d(x):
    x[x>0]=1
    x[x<0]=0
    return x

def softmax(x):
    num = np.exp(x)
    den = np.sum(np.exp(x), axis=1).reshape(-1,1)
    return num/den

def mseloss(x, y):
    mse = (1/len(x))*(sum((x-y)**2))
    return mse

def onehot(Y):
    true = np.zeros((len(Y), 10), np.float64)
    true[range(true.shape[0]), Y] = 1
    return(true)

#===============================================================================
# Feedforward and Backpropogation Functions
#===============================================================================

def feedforward(X, W1, B1, W2):

    A1 = X

    # Layer 1
    Z1 = np.matmul(A1, W1) + B1
    A2 = relu(Z1)

    # Layer2
    Z2 = np.matmul(A2, W2)
    A3 = softmax(Z2)

    return A1, Z1, A2, Z2, A3

def backprop(lr, batch_size, Y, W1, B1, W2, A1, A2, A3, Z1):


    # Find batch error.
    Y = onehot(Y)
    err_L2 = A3 - Y

    # Calculate gradients.
    dW2 = 1/batch_size * np.matmul(A2.T, err_L2)

    err_L1 = np.matmul(err_L2, W2.T)*relu_d(Z1)
    dW1 = 1/batch_size * np.matmul(A1.T, err_L1)

    dB1 = np.mean(err_L1, axis=0)

    # Update weights.
    W1 = W1 - lr * dW1
    B1 = B1 - lr * dB1
    W2 = W2 - lr * dW2

    return W1, B1, W2

def run_epoch(X, Y, batch_size, lr, W1, B1, W2):

    epoch_loss = 0.0

    # Find number of batches in the epoch.
    num_batches = int(len(X) / batch_size)

    for i in range(num_batches):

        # Random sample batch_size observations without replacement.
        obs = np.random.choice(len(X), batch_size, replace = False)
        X = X[obs]
        Y = Y[obs]

        # Forward propogation.
        A1, Z1, A2, Z2, A3 = feedforward(X, W1, B1, W2)

        # Backward propogation.
        W1, B1, W2 = backprop(lr, batch_size, Y, W1, B1, W2, A1, A2, A3, Z1)

        # Calculate batch loss.
        batch_loss = sum(mseloss(A3, onehot(Y)))
        epoch_loss += batch_loss

        if i % 10 == 0:
            batch_acc = sum(np.argmax(A3, axis=1) == Y) / batch_size
            print(batch_acc)

    print(epoch_loss)
    return W1, B1, W2

# Initialize layer weights. 
k = math.sqrt(1/(28*28))
W1 = np.random.uniform(low = -k, high = k, size = (28*28,128)) 
B1 = np.full(shape = (128,), fill_value = 0.01)
W2 = np.random.uniform(low = -k, high = k, size = (128,10)) 

W1, B1, W2 = run_epoch(train_imgs, train_lbls, 32, 0.01, W1, B1, W2)

