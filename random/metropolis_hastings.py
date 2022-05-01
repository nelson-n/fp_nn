
#-------------------------------------------------------------------------------
# metropolis_hastings.py written by nelson-n 2022-04-30
#
# Implements Metropolis-Hastings first using the canonical package PyMC3 and 
# then by scratch.  
#-------------------------------------------------------------------------------

# The Metropolis-Hastings method is a Markov Chain Monte Carlo approach to 
# obtaining a sequence of random samples from a distribution from which direct
# sampling is difficult. This sequence can then be used to approximate the 
# distribution. I.e. you have observed some data that may be difficilt to 
# observe, and you want to know what parameters have given rise to this data.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import tqdm

# Generate sample data with mean 1.5 sd 1.2.
mean = 1.5
sd = 1.2

X = np.random.normal(mean, sd, size = 1000)

# Verify true parameters. 
X.mean()
X.std()

#===============================================================================
# Metropolis-Hastings Implementation with PyMC3
#===============================================================================

import pymc3 as pm

# Implementing Metropolis-Hastings where our normal prior is that both the mean
# and the sd are distributed with mean 2 sd 2.
with pm.Model() as model:

  mean = pm.Normal("mean", mu = 2, sigma = 2)
  std = pm.Normal("std", mu = 2, sigma = 2)
  x = pm.Normal("X", mu = mean, sigma = std, observed = X)

  step = pm.Metropolis()
  trace = pm.sample(5000, step = step)

# View samples.
trace['mean']
trace['std'].mean()

#===============================================================================
# Metropolis-Hastings Implementation from Scratch
#===============================================================================

# To sample, MH randomly selects samples from a space. This is done using a 
# proposal distribution centered at the currently accepted sample with a 
# standard deviation of 0.5, 0.5 is a hyperparameter and as it is increased
# the algorithm will search farther from the currently accepted sample.

# This function takes the currently accepted mean and sd and returns a proposal
# for both.
def get_proposal(mean_current, std_current, proposal_width = 0.5):

    return np.random.normal(mean_current, proposal_width), \
           np.random.normal(std_current, proposal_width)

# Next are the functions that accept or reject the proposed mean and sd. 
# Acceptance or rejection is a function of the prior and the likelihood.

# The first function calculates the probability of the proposal coming from the 
# prior. This is the likelihood that the proposed mean and std comes from the 
# prior distribution.
def prior(mean, std, prior_mean, prior_std):

        return st.norm(prior_mean[0], prior_mean[1]).pdf(mean)* \
               st.norm(prior_std[0], prior_std[1]).pdf(std)

# The second function calculates the likelihood that the data we saw comes from
# the proposed distribution.
def likelihood(mean, std, data):
    return np.prod(st.norm(mean, std).pdf(X))

# Full function for accepting or rejecting proposed sample.
def accept_proposal(mean_proposal, std_proposal, mean_current, std_current,
	prior_mean, prior_std, data):
	
    # Find prior and likelihood for current proposition.
    prior_current = prior(mean_current, std_current, prior_mean, prior_std)
    likelihood_current = likelihood(mean_current, std_current, data)

    # Find prior and likelihood for last proposition.
    prior_proposal = prior(mean_proposal, std_proposal, prior_mean, prior_std)
    likelihood_proposal = likelihood(mean_proposal, std_proposal, data)

    # Scale prior and likelihood for both proposals.
    return (prior_proposal * likelihood_proposal) / (prior_current * likelihood_current)

# Full function for generating the posterior.
def get_trace(mean_prior, std_prior, data, samples = 5000):

    mean_current = mean_prior
    std_current = std_prior

    trace = {
        "mean": [mean_current],
        "std": [std_current]
    }

    for i in tqdm(range(samples)):

        mean_proposal, std_proposal = get_proposal(mean_current, std_current)

        acceptance_prob = accept_proposal(mean_proposal, std_proposal, mean_current, 
            std_current, [mean_prior, std_prior], [mean_prior, std_prior], data)

        if np.random.rand() < acceptance_prob:
            mean_current = mean_proposal
            std_current = std_proposal

        trace['mean'].append(mean_current)
        trace['std'].append(std_current)

    return trace

# Run hand coded Metropolis-Hastings.
get_trace(mean_prior = 2, std_prior = 2, data = X)

