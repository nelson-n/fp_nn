
#-------------------------------------------------------------------------------
# ml_nn_problems.py written by nelson-n 2022-01-09 
# 
# Solves simple problems related to machine learnings and neural networks.
#-------------------------------------------------------------------------------

import math
import numpy as np

#===============================================================================
# Problem 1
#===============================================================================

# Given the equation:
# f(x,y) = x^2 - 2siny with x in [-1, 1] and y in [0, pi].
# Find the optima of f and the location of the optima.
def obj_func(x, y):
    return x**2 - 2*math.sin(y)

# Gradient of the function = [2x, -2cosy].
# Begin with the initial guess of [1/2, pi/2].

# Set fixed learning rate.
lr = 0.01

# Function for performing gradient descent given an initial x, y and number
# of iterations.
def grad_desc(x, y, iters):

    for i in range(iters):
        x = x - lr * (2*x)
        y = y - lr * (-2*math.cos(y))

    minima = obj_func(x, y)
    return x, y, minima

grad_desc(1/2, math.pi/2, iters = 1000)










