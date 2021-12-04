## fp_nn: Neural Networks built from first principles in NumPy.

### feedforward_nn
* **feedforward_nn.py** train a feedforward neural network for classifying MNIST
handwritten digits.
* MNIST data available at: [MNIST Database](http://yann.lecun.com/exdb/mnist/)

### rnn
* Trains recurrent neural network for sentence completion/prediction using the
text of Leo Tolstoy's *War and Peace*.
* **process_text.py** converts the raw *War and Peace* text file into tokenized,
sentence level numpy arrays for use in rnn.py. 
* **rnn.py** trains a token level RNN. Heavily inspired by: 
[pangolulu/rnn-from-scratch](https://github.com/pangolulu/rnn-from-scratch)
* **char_rnn.py** trains a character level RNN. This script is a clone of 
[karpathy/min-char-rnn.py](https://gist.github.com/karpathy/d4dee566867f8291f086) 
which tests min-char-rnn.py on the *War and Peace* data and adds more verbose 
documentation for my own benefit.
* My preferred RNN visualization is the following:
![](./imgs/rnn_graph.png)

### mle
* Maximum likelihood estimation of a normal distribution from scratch.
* **mle.R** includes a density function, negative log likelihood function,
and a naive optimization function for performing MLE.
* This is benchmarked against MLE performed with Nelder-Mead simplex optimization.
An implementation of Nelder-Mead optimization from scratch is available at:
[nicolaivicol/nelder-mead-R](https://github.com/nicolaivicol/nelder-mead-R)
* Likelihood calculated as:
![](.imgs/likelihood.png)

