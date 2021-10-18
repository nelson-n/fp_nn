
#-------------------------------------------------------------------------------
# char_rnn.py is a clone of karpathy/min-char-rnn.py which runs the character
# level RNN script on War and Peace text as well as adds more verbose comments
# for my own understanding.
#-------------------------------------------------------------------------------

import numpy as np

#===============================================================================
# Text Pre-Processing
#===============================================================================

# Open War and Peace .txt file.
data = open('war-and-peace.txt', 'r').read() # should be simple plain text file

# Find all characters that show up in the .txt file.
chars = list(set(data))

# Find number of characters in corpus and number of unique characters.
data_size, vocab_size = len(data), len(chars)

# Create dictionary for converting between numeric and text form.
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#===============================================================================
# Hyperparameters and Model Parameters
#===============================================================================

# Set the size of the hidden layer of neurons.
hidden_size = 100 # size of hidden layer of neurons

# Set the number of steps that the RNN will be unrolled for.
seq_length = 25 

learning_rate = 1e-1

# Wxh is the input layer weights (100, 104) which will be dotted with a (104, 1)
# one-hot input vector which results in a vector of size (100, 1).
Wxh = np.random.randn(hidden_size, vocab_size)*0.01

# Whh is the hidden weights (100, 100) which will be dotted with a hidden state
# vector (100, 1) which results in a vector of size (100, 1). This is then added
# with the vector produced by the input layer weights above and then passed
# through the tanh activation function, resulting in a vector of size (100, 1).
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden

# Why is the output (104, 100) which is dotted with the vector (100, 1) described
# by the process above, resulting in a (104, 1) output layer which is then passed
# through softmax to find the probabilities of the next character in the sequence.
Why = np.random.randn(vocab_size, hidden_size)*0.01 

# Hidden layer bias (100).
bh = np.zeros((hidden_size, 1)) 

# Output layer bias (104).
by = np.zeros((vocab_size, 1)) 

#===============================================================================
# Define Loss Function (Forward Pass and Backward Pass)
#===============================================================================

def lossFun(inputs, targets, hprev):

  # Initialize blank input (xs), hidden state (hs), output layer (ys), and 
  # activated probabilities (ps).
  xs, hs, ys, ps = {}, {}, {}, {}

  # Set hidden state as whatever initial hidden state was passed via hprev argument.
  hs[-1] = np.copy(hprev)

  loss = 0

  # Forward pass.
  # For each character in a string of characters:
  for t in range(len(inputs)):

    # Create one-hot vector for input character encoding.
    xs[t] = np.zeros((vocab_size,1)) 
    xs[t][inputs[t]] = 1

    # Combine input and hidden layer, calculate tanh activation.
    # Hidden state = tanh(dot(Wx, x) + dot(Wh, hs[t-1]) + bh)
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) 

    # Calculate output layer (not yet activated).
    ys[t] = np.dot(Why, hs[t]) + by 

    # Apply softmax to output layer (this length 104 vector represents the 
    # probabilities of the next characters)
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))

    # Calculate cross entropy loss based on how far softmax output is from 
    # a perfect prediction.
    loss += -np.log(ps[t][targets[t],0]) 

  # Backward pass, compute gradients.

  # Initiate weight and bias derivates as array of zeroes.
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)

  dhnext = np.zeros_like(hs[0])

  # Backward pass in reverse order.
  for t in reversed(range(len(inputs))):

    # dy is the softmax prediction vector.
    dy = np.copy(ps[t])

    # Subtract 1 from the true character.
    dy[targets[t]] -= 1     

    # Derivate of weights is the dot of the downstream layer error (dy) and the 
    # transposed current layer inputs hs[t].
    dWhy += np.dot(dy, hs[t].T)

    # Add the downstream layer error (dy) to the derivate of the current layer bias.
    dby += dy

    # Keep backpassing through combination layer and tanh activation by dotting
    # the transposed layer weights and the activation dy.
    dh = np.dot(Why.T, dy) + dhnext 
    
    # Backprop through tanh nonlinearity.
    dhraw = (1 - hs[t] * hs[t]) * dh 

    # Add to hidden layer bias derivative.
    dbh += dhraw

    # Derivative of weights is the dot of the downstream layer error and the
    # transposed current layer inputs.
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)

    # Calculate hidden layer error by dotting the transposed layer weights with
    # the activation. This hidden layer error is used as we iterate through the 
    # inputs, running the backward pass.
    dhnext = np.dot(Whh.T, dhraw)

  # Limit the derivative values to prevent exploding gradients.
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) 

  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

#===============================================================================
# Test RNN
#===============================================================================

# Pass a memory state (h), a first character (seed_ix), and a length of
# characters (n) through the RNN in a forward pass. 
def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """

  # Create one-hot vector for seed_ix character input.
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []

  # Forward pass.
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

#===============================================================================
# Train RNN
#===============================================================================

# Prepare weights and biases for training, m stands for memory.
# p = the total number of characters trained through.
# n = the total number of iterations of length seq_length that have been passed
# through.
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) 

# Set loss at iteration 0.
smooth_loss = -np.log(1.0/vocab_size)*seq_length 

# Continuously train model.
while True:

  # The model is continuously trained throughout the whole corpus, once it reaches
  # the end of the corpus RNN memory is reset and the training start from the 
  # beginning of the data (p = 0).
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data

  # Convert inputs (the current character to seq_length characters ahead) to integers.
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]

  # Convert targets (the current character+1 to seq_length+1 characters ahead) 
  # to integers.
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # Every 10000 characters, have the RNN produce a token to see if it has become
  # coherant. 
  if n % 10000 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('\n \n \n')
    print('sample text: ', txt)
    print('\n')

  # Forward seq_length characters through the net and fetch gradients.
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  # Monitor progress.
  if n % 10000 == 0: print('iter %d, loss: %f' % (n, smooth_loss))
  
  # Adagrad parameter update.
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam

    # Adagrad update.
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) 

  # Increment p and n.
  p += seq_length 
  n += 1

