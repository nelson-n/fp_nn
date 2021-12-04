
#-------------------------------------------------------------------------------
# mle.R written by lucius-verus-fan 2021-11-29
#
# Maximum likelihood estimates of a normal distribution from scratch.
#-------------------------------------------------------------------------------

set.seed = 42

# Generate 10000 values from a N(1, 2) distribution.
sample <- rnorm(10000, mean = 1, sd = 2)

# Given a statistical model of probability density and a sample of i.i.d 
# observations, the likelihood of observing the whole sample is defined as the 
# product of the probability densities of the individual values. The goal of MLE
# is to find the parameters of the statistical model that maximize the 
# likelihood of observing the whole sample. This is done by minimizing negative
# log likelihood (NLL). MLE converges asymptotically to the true parameters of 
# the population as the sample size increases.

# Normal distribution function.
normal <- function(x, mu, sigma) {
  (1 / (sqrt(2 * pi * sigma^2))) * (exp((-(x - mu)^2) / (2 * sigma^2)))
}

# Negative log likelihood function.
nll <- function(pars, data) {
  mu = pars[1]
  sigma = pars[2]
  
  -sum(log(normal(x = data, mu, sigma)))
}

# Super naive optimization function written from scratch.
# Note* this is not a good optimazation function and it occasionally gets stuck at
# local minima.

# n_iters = number of iterations through the search process.
# init.mu = original guess of the mean of the distribution.
# init.sigma = original guess of the standard deviation of the distribution.
# mu.scope = the maximum and minimum mu values to test (mu scope = init.mu +- mu.scope)
# sigma.scope = the maximum and minimum sigma values to test (sigma scope = init.sigma +- sigma.scope)
# granularity = number of values between the mu and sigma scopes to test.
# scope.shrinkage = the amount the scope window is shrunk each iteration.

mle <- function(x, n_iters, init.mu, init.sigma, mu.scope, sigma.scope, granularity, scope.shrinkage) {
  
  test.mu <- init.mu
  test.sigma <- init.sigma
  
  for (n in 1:n_iters) {
    
    # Create vectors of test mu and sigma values.
    pars <- cbind(
      seq(
        from = (test.mu - mu.scope),
        to = (test.mu + mu.scope),
        length.out = granularity
      ),
      seq(
        from = (test.sigma - sigma.scope),
        to = (test.sigma + sigma.scope),
        length.out = granularity
      )
    )
    
    # Calculate NLL for each mu and sigma value.
    nll_score <- vector(mode = "numeric", length = nrow(pars))
    
    for (i in 1:nrow(pars)) {
      nll_score[[i]] <- nll(pars[i, ], x)
    }
    
    # Find new optimal mu and sigma
    test.mu <- pars[which.min(nll_score), 1]
    test.sigma <- pars[which.min(nll_score), 2]
    
    # Apply search window shrinkage.
    mu.scope <- mu.scope * scope.shrinkage
    sigma.scope <- sigma.scope * scope.shrinkage
    
  }
  
  return(c(test.mu, test.sigma))
  
}

# Run MLE.
mle(x = sample, n_iters = 1, init.mu = 0, init.sigma = 1, mu.scope = 5, 
    sigma.scope = 5, granularity = 100, scope.shrinkage = 0.75)

# Test performance relative to MLE that uses the dnorm function and 
# Nelder-Mead simplex optimization.
benchmark_nll = function(pars, data) {
  
  mu = pars[1]
  sigma = pars[2]
  -sum(dnorm(x = data, mean = mu, sd = sigma, log = TRUE))
}

benchmark_mle <- optim(par = c(mu = 1, sigma = 1), fn = benchmark_nll, data = sample,
                       control = list(parscale = c(mu = 1, sigma = 1)))

benchmark_mle$par
