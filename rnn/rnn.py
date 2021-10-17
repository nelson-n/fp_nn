
# -------------------------------------------------------------------------------
# rnn.py written by lucius-verus-fan 2021-10-10.
#
# Creates recurrent neural network from scratch in numpy.
# Based heavily on pangolulu/rnn-from-scratch.
# -------------------------------------------------------------------------------

import numpy as np
from process_text import process_text

# Create class for performing matmuls.
class matmul:
    # Forward pass is simply a dot product of weights and input matrix.
    def forward(self, W, x):
        return np.dot(W, x)

    def backward(self, W, x, dz):

        # Returns derivative of the weights and the layer error. Obtaining the 
        # derivative of the weights is the objective of backprop, and obtaining
        # the layer error of downstream errors is necessary for computing 
        # upstream derivatives.

        # The derivative of the weights is the dot of the previous activation
        # (dz) transposed and input matrix x (errors).
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))

        # The layer error is the dot of the transposed weights and the previous
        # activation (dz).
        dx = np.dot(np.transpose(W), dz)
        return dW, dx

# Create class for performing matrix addition.
class matadd:
    def forward(self, x1, x2):
        return x1 + x2

    # Bias error is simply the previous activation (dz).
    def backward(self, x1, x2, dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1, dx2

# Create class for sigmoid activation function.
# Sigmoid activation does not actually get used.
class sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - output) * output * top_diff

# Create class for tanh activation function.
class tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff

# Create class for softmax activation function.
class softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores)

    def loss(self, x, y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self, x, y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs

# Create class for building RNN layers.
class RNNLayer:
    def forward(self, x, prev_s, U, W, V):

        # Forward pass current state. U is the current token weights and 
        # x_t is the current token.
        self.mulU = matmul.forward(U, x)

        # Forward pass last state. W is the last token weights and s_t-1
        # is the last token.
        self.mulW = matmul.forward(W, prev_s)

        # Add forward passes together. Last token and current token matmuls are
        # combined.
        self.add = matadd.forward(self.mulU, self.mulW)

        # Pass forward passes through tanh activation.
        self.s = tanh.forward(self.add)

        # Matmul activation and V. V is the combined activation weights and s_t
        # is the combined activation. Softmax is later applied to this output
        # to predict the token.
        self.mulV = matmul.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):

        # Pass forward.
        self.forward(x, prev_s, U, W, V)

        # Calculating derivative of s_t and V layer:
        # matmul.backward returns the derivative of the weights and the layer 
        # error. dmulv (the softmax token prediction output error), the combined
        # activation s_t, and the combined activation weights V are passed as 
        # arguments.
        dV, dsv = matmul.backward(V, self.s, dmulv)

        # s_t layer error is calculated as layer error difference from 0.
        ds = dsv + diff_s

        # Calculating derivative of x_t and s_t-1 combination layer.
        # Backward pass through tanh activation and combination.
        dadd = tanh.backward(self.add, ds)
        dmulw, dmulu = matadd.backward(self.mulw, self.mulu, dadd)

        # Calculating derivatives of s_t-1 * W matmul and x_t * U matmul.
        dW, dprev_s = matmul.backward(W, prev_s, dmulw)
        dU, dx = matmul.backward(U, x, dmulu)

        # Return updated derivatives and error for s_t-1 layer.
        return (dprev_s, dU, dW, dV)

# Create model class for initializing weights, forward passes, back passes, etc.
class Model:

    # Initialize weights of U, W, and V. These are arrays of shape hidden_dim,
    # x word_dim that are initialized with random values between -1/sqrt(n) to
    # 1/sqrt(n) where n is the input dimension.
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

        # hidden_dim is the number of tokens to look back on.
        # word_dim is the number of tokens in the vocabulary.
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))

    def forward_propagation(self, x):
        # x = the length of the sentence in tokens, this is the total number of timesteps.
        timesteps = len(x)
        layers = []

        # The original prev_s hidden state is set to all zeroes. For each sentence,
        # we will not use the sentence before to predict subsequent tokens.
        prev_s = np.zeros(self.hidden_dim)

        # For each timestep in a sentence.
        for t in range(timesteps):

            # Create a blank RNN pass for the current token.
            layer = RNNLayer()

            # Create a one-hot vector the size of word_dim where a 1 represents
            # the position of the current word in the sentence.
            input_vec = np.zeros(self.word_dim)
            input_vec[x[t]] = 1

            # Perform forward pass.
            layer.forward(x, prev_s, self.U, self.W, self.V)

            # Extract hidden state from forward pass. The hidden state is
            # the tanh activation of the combined layers.
            prev_s = layer.s

            # Create list of all layers from a sentence.
            layers.append(layer)

        return layers

    def predict(self, x):

        softmax = softmax()

        # Perform forward pass on all tokens in the sentence.
        layers = self.forward_propagation(x)

        # Apply softmax to all tokens in the sentence.
        return [np.argmax(softmax.predict(layer.mulv)) for layer in layers]

    # Cross-entropy loss function.
    def calculate_loss(self, x, y):

        # Confirm input and prediction sentences are the same length.
        assert len(x) == len(y)
        softmax = softmax()
        layers = self.forward_propagation(x)
        loss = 0.0

        for i, layer in enumerate(layers):
            loss += softmax.loss(layer.mulv, y[i])

        # Convert total loss to the average loss for each word.
        return loss / float(len(y))

    # Calculate loss for the entire corpus.
    def calculate_total_loss(self, X, Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i], Y[i])
        return loss / float(len(Y))

    # Backpropogation through time that returns the gradients dL/dW, dL/dU, dL/dV
    def bptt(self, x, y):

        # Forward pass the sentence x.
        assert len(x) == len(y)
        output = softmax()
        layers = self.forward_propagation(x)

        # Initialize gradients as 0.
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        timesteps = len(layers)

        # Initialize hidden state as all zeros.
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)

        for t in range(0, timesteps):

            # Find softmax (token prediction) error as the difference between
            # the predicted token and the true token y[t].
            dmulv = output.diff(layers[t].mulv, y[t])

            # Create one hot vector with the x token = 1.
            input = np.zeros(self.word_dim)
            input[x[t]] = 1

            # Backpropagate.
            dprev_s, dU_t, dW_t, dV_t = layers[t].backward(
                input, prev_s_t, self.U, self.W, self.V, diff_s, dmulv)

            # Find the hidden state.
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.word_dim)

            # To combat the vanishing gradient problem, we only let the RNN
            # look back at the past bptt_truncate tokens when calculating 
            # the gradients.
            for i in range(t-1, max(-1, t-self.bptt_truncate-1), -1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(
                    self.hidden_dim) if i == 0 else layers[i-1].s
                dprev_s, dU_i, dW_i, dV_i = layers[i].backward(
                    input, prev_s_i, self.U, self.W, self.V, dprev_s, dmulv)
                dU_t += dU_i
                dW_t += dW_i

            # Update weights.
            dV += dV_t
            dU += dU_t
            dW += dW_t
        return (dU, dW, dV)

    # Perform gradient descent on one sentence.
    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.bptt(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW

    # Function to train over all sentences.
    def train(self, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        num_examples_seen = 0
        losses = []

        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X, Y)
                losses.append((num_examples_seen, loss))

                # If loss is increasing, increase learning rate.
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                sys.stdout.flush()

            # Passing through all sentences in the corpus.
            for i in range(len(Y)):
                self.sgd_step(X[i], Y[i], learning_rate)
                num_examples_seen += 1
        return losses


# Pull in War and Peace text.
X_train, Y_train = process_text()

# Set number of words in vocabulary to 10000.
word_dim = 10000

# Set dimension of hidden layer to 100.
hidden_dim = 100

# Test one pass through rnn with one sentence.
rnn = Model(word_dim, hidden_dim)
rnn.sgd_step(X_train[12], Y_train[12], learning_rate=0.005)

# Test training with 1000 sentences.
training = rnn.train(X_train[:100], Y_train[:100], learning_rate=0.05, nepoch=10)

