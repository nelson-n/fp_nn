
#-------------------------------------------------------------------------------
# ols_logit_sgd.py written by nelson-n 2022-01-24
# 
# Estimates OLS and logit coefficients using stochastic gradient descent.
#-------------------------------------------------------------------------------

import math
import random

#==============================================================================
# OLS Using Stochastic Gradient Descent
#==============================================================================

# Function for making an OLS prediction (yhat) given a matrix of covariates X,
# a vector of true values y, and a list of coefficients.
def predict_ols(Xy, coefficients):

	yhat = coefficients[0]

	for i in range(len(Xy)-1):
		yhat += coefficients[i + 1] * Xy[i]

	return yhat

# Function for updating OLS coefficents using online SGD. Coefficients are
# initiated to equal 0, one sample (row) of the training set is randomly
# selected, the prediction error is calculated and scaled by the learning rate, 
# the intercept and coefficients are updated to minimize error.
def coefficients_sgd(Xy_train, lr, epochs):

	coef = [0.0 for i in range(len(Xy_train[0]))]

	for epoch in range(epochs):

		for row in random.sample(Xy_train, 1):
			yhat = predict_ols(row, coef)
			error = yhat - row[-1]

			coef[0] = coef[0] - lr * error
			for i in range(len(row)-1):
				coef[i + 1] = coef[i+1] - lr * error * row[i]

	return coef

# Given a training dataset and a testing dataset, estimate coefficients via
# SGD on the training dataset and use these coefficients to make OLS predictions
# on the testing dataset.
def linear_regression_sgd(train, test, lr, epochs):

	predictions = list()
	coef = coefficients_sgd(train, lr, epochs)

	for row in test:
		yhat = predict_ols(row, coef)
		predictions.append(yhat)

	return(predictions)

# Function for calculating root mean squared error.
def rmse(actual, predicted):

	sum_error = 0.0

	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)

	mean_error = sum_error / float(len(actual))

	return math.sqrt(mean_error)

# Synthetic dataset with expected intercept of 0 and coefficient of 1.
dataset = [[i, 2*i+random.gauss(0, 0.1)] for i in range(0, 100)]
lr = 0.00001
epochs = 1000
coefficients_sgd(dataset, lr, epochs)

# Train and evaluate OLS coefficients on test set.
train = [[i, 2*i+random.gauss(0, 0.1)] for i in range(0, 100)]
test = [[i, 2*i+random.gauss(0, 0.1)] for i in range(102, 122)]
lr = 0.0001
epochs = 1000
predicted = linear_regression_sgd(train, test, lr, epochs)
actual = [t[-1] for t in test]
rmse(actual, predicted)

#==============================================================================
# Logit Using Stochastic Gradient Descent
#==============================================================================

# Testing SGD using logit instead of OLS. 

def predict_logit(Xy, coefficients):

	yhat = coefficients[0]

	for i in range(len(Xy)-1):
		yhat += coefficients[i + 1] * Xy[i]

	return 1.0 / (1.0 + math.exp(-yhat))
 
def coefficients_sgd(Xy_train, lr, epochs):

	coef = [0.0 for i in range(len(Xy_train[0]))]

	for epoch in range(epochs):

		for row in random.sample(Xy_train, 1):
			yhat = predict_logit(row, coef)
			error = row[-1] - yhat

			coef[0] = coef[0] + lr * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + lr * error * yhat * (1.0 - yhat) * row[i]

	return coef
 
def logistic_regression(train, test, lr, epochs):

	predictions = list()
	coef = coefficients_sgd(train, lr, epochs)

	for row in test:
		yhat = predict_logit(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)

	return(predictions)

def accuracy_metric(actual, predicted):

	correct = 0

	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1

	return correct / float(len(actual)) * 100.0

# Synthetic dataset.
dataset = [2*[random.gauss(0, 0.1)] for i in range(0, 500)]
dataset = [[t[0], max(t[0]/abs(t[0]), 0.0)] for t in dataset]
lr = 0.1
epochs = 200
coef = coefficients_sgd(dataset, lr, epochs)
print(coef)

# Train and evaluate logit coefficients on test set.
train = [2*[random.gauss(0, 1)] for i in range(0, 500)]
train = [[t[0],max(t[0]/abs(t[0]), 0.0)] for t in train]
test = [2*[random.gauss(0, 1)] for i in range(0, 100)]
test = [[t[0],max(t[0]/abs(t[0]), 0.0)] for t in test]
lr = 0.1
epochs = 200
predicted = logistic_regression(train, test, lr, epochs)
actual = [t[-1] for t in test]
accuracy_metric(actual, predicted)

