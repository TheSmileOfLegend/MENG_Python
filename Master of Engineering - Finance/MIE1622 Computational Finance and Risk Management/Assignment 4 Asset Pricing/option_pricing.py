from numpy import *
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Pricing a European option using Black-Scholes formula and Monte Carlo simulations 
# Pricing a Barrier option using Monte Carlo simulations

S0 = 100     # spot price of the underlying stock today
K = 105      # strike at expiry
mu = 0.05    # expected return
sigma = 0.2  # volatility
r = 0.05     # risk-free rate
T = 1.0      # years to expiry
Sb = 110     # barrier


# Complete the following functions
def BS_european_price(S0, K, T, r, sigma):
  # --------- Insert your code here --------- #
  d1 = 1 / (sigma * sqrt(T)) * (log(S0 / K) + (r + sigma ** 2 / 2) * T)
  d2 = d1 - sigma * sqrt(T)
  c = norm.cdf(d1) * S0 - norm.cdf(d2) * K * exp(- r * T)
  p = norm.cdf(-d2) * K * exp(-r * T) - norm.cdf(-d1) * S0
  return c, p

def MC_european_price(S0, K, T, r, mu, sigma, numSteps, numPaths):
  # --------- Insert your code here --------- #  
  paths = np.zeros((numSteps + 1, numPaths))
  
  # dT is the time increment (in years)
  dT = T / numSteps
  
  # Vector of paths will store realizations of the asset price
  # First asset price is the initial price
  paths[0] = [S0] * numPaths

  # Generate paths
  for iPath in range(numPaths):
      for iStep in range(numSteps):
          paths[iStep + 1, iPath] = paths[iStep, iPath] * np.exp((mu - 0.5 * sigma ** 2) * dT + sigma * np.sqrt(dT) * np.random.normal(0,1))
    
  PutPayoffT = np.maximum(K - paths[numSteps,:], 0)
  CallPayoffT = np.maximum(paths[numSteps,:] - K, 0)
  p = np.mean(PutPayoffT) * np.exp(-r * T)
  c = np.mean(CallPayoffT) * np.exp(-r * T)
  return c, p

def MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma, numSteps, numPaths):
  # --------- Insert your code here --------- #
  paths = np.zeros((numSteps + 1, numPaths))
  
  # dT is the time increment (in years)
  dT = T / numSteps
  
  # Vector of paths will store realizations of the asset price
  # First asset price is the initial price
  paths[0] = [S0] * numPaths
  knockin_flags = np.zeros(numPaths)
  # Generate paths
  for iPath in range(numPaths):
      for iStep in range(numSteps):
          paths[iStep + 1, iPath] = paths[iStep, iPath] * np.exp((mu - 0.5 * sigma ** 2) * dT + sigma * np.sqrt(dT) * np.random.normal(0,1))
          if paths[iStep + 1, iPath] >= Sb:
          	knockin_flags[iPath] = 1

  PutPayoffT = np.maximum(K - paths[numSteps,:], 0)
  CallPayoffT = np.maximum(paths[numSteps,:] - K, 0)

  for iPath in range(numPaths):
      if not knockin_flags[iPath]:
          PutPayoffT[iPath] = 0
          CallPayoffT[iPath] = 0
          

  p = np.mean(PutPayoffT) * np.exp(-r * T)
  c = np.mean(CallPayoffT) * np.exp(-r * T)
  return c, p

# Define variable numSteps to be the number of steps for multi-step MC
# numPaths - number of sample paths used in simulations

numSteps = 10;
numPaths = 1000000;

# Implement your Black-Scholes pricing formula
call_BS_European_Price, putBS_European_Price = \
  BS_european_price(S0, K, T, r, sigma)

np.random.seed(10)
# Implement your one-step Monte Carlo pricing procedure for European option
# (7.190, 7.600) 1k paths
# (7.953, 7.774) 10k paths
# (7.974, 7.916) 100k paths
# (8.012, 7.900) 1M paths
callMC_European_Price_1_step, putMC_European_Price_1_step = \
  MC_european_price(S0, K, T, r, mu, sigma, numSteps = 1, numPaths = 1000000)

np.random.seed(5)
# Implement your multi-step Monte Carlo pricing procedure for European option
# (8.195, 7.788) 10k paths
# (8.078, 7.897) 100k paths
# (8.025, 7.893) 1M paths
callMC_European_Price_multi_step, putMC_European_Price_multi_step = \
  MC_european_price(S0, K, T, r, mu, sigma, numSteps = 12, numPaths = 1000000)

np.random.seed(6)
# Implement your multi-step Monte Carlo pricing procedure for European option
# (7.901, 0.0) 10k paths
# (7.741, 0.0) 100k paths
# (7.810, 0.0) 1M paths
# Implement your one-step Monte Carlo pricing procedure for Barrier option
callMC_Barrier_Knockin_Price_1_step, putMC_Barrier_Knockin_Price_1_step = \
  MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma, numSteps = 1, numPaths = 1000000)

np.random.seed(3)
# Implement your multi-step Monte Carlo pricing procedure for European option
# (7.952, 1.297) 10k paths
# (8.037, 1.285) 100k paths
# (7.975, 1.272) 1M paths
# Implement your multi-step Monte Carlo pricing procedure for Barrier option
callMC_Barrier_Knockin_Price_multi_step, putMC_Barrier_Knockin_Price_multi_step = \
  MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma, numSteps = 12, numPaths = 1000000)

print('Black-Scholes price of an European call option is ' + str(call_BS_European_Price))
print('Black-Scholes price of an European put option is ' + str(putBS_European_Price))
print('One-step MC price of an European call option is ' + str(callMC_European_Price_1_step)) 
print('One-step MC price of an European put option is ' + str(putMC_European_Price_1_step)) 
print('Multi-step MC price of an European call option is ' + str(callMC_European_Price_multi_step)) 
print('Multi-step MC price of an European put option is ' + str(putMC_European_Price_multi_step)) 
print('One-step MC price of an Barrier call option is ' + str(callMC_Barrier_Knockin_Price_1_step)) 
print('One-step MC price of an Barrier put option is ' + str(putMC_Barrier_Knockin_Price_1_step)) 
print('Multi-step MC price of an Barrier call option is ' + str(callMC_Barrier_Knockin_Price_multi_step)) 
print('Multi-step MC price of an Barrier put option is ' + str(putMC_Barrier_Knockin_Price_multi_step))

# Plot results
# --------- Insert your code here --------- #
def GRWPaths(initPrice, mu, sigma, T, numSteps, numPaths):
    
    paths = np.zeros((numSteps + 1, numPaths))
    
    # dT is the time increment (in years)
    dT = T / numSteps
    
    # Vector of paths will store realizations of the asset price
    # First asset price is the initial price
    paths[0] = [initPrice] * numPaths
 
    # Generate paths
    for iPath in range(numPaths):
        for iStep in range(numSteps):
            paths[iStep + 1, iPath] = paths[iStep, iPath] * np.exp((mu - 0.5 * sigma ** 2) * dT 
                                                                   + sigma * np.sqrt(dT) * np.random.normal(0,1))
    
    prices = [paths[:,i] for i in range(numPaths)]
    times = [dT * iPath for iPath in range(numSteps + 1)]
    return prices, times

np.random.seed(5)

# generate and plot geometric random walks for one-step and multi-step
plot_prices_one_step, plot_times_one_step = GRWPaths(S0, mu, sigma, T, numSteps = 1, numPaths = 1)
plot_prices_multi_step, plot_times_multi_step = GRWPaths(S0, mu, sigma, T, numSteps = 12, numPaths = 1)
plt.plot(plot_times_one_step, plot_prices_one_step[0], linewidth=2)
plt.plot(plot_times_multi_step, plot_prices_multi_step[0], linewidth=2)

# plot barrier price and strike price
plt.plot([0, 1], [Sb, Sb], linewidth=2, linestyle='dashed')
plt.plot([0, 1], [K, K], linewidth=2, linestyle='dashed')
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.legend(['one-step GRW', 'multi-step GRW', 'Barrier Price', 'Strike Price'])
plt.title('Illustration of Monte Carlo Pricing Procedure', fontsize=14)
plt.show()

np.random.seed(3)
# multi-step MC Barrier with modified volatility
callMC_Barrier_Knockin_Price_multi_step_decreased_volatility, putMC_Barrier_Knockin_Price_multi_step_decreased_volatility= \
  MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma * 0.9, numSteps = 12, numPaths = 1000000)
callMC_Barrier_Knockin_Price_multi_step_increased_volatility, putMC_Barrier_Knockin_Price_multi_step_increased_volatility= \
  MC_barrier_knockin_price(S0, Sb, K, T, r, mu, sigma * 1.1, numSteps = 12, numPaths = 1000000)
print('Multi-step MC price of an Barrier call option with 10% decreased volatility is ' + str(callMC_Barrier_Knockin_Price_multi_step_decreased_volatility)) 
print('Multi-step MC price of an Barrier put option with 10% decreased volatility is ' + str(putMC_Barrier_Knockin_Price_multi_step_decreased_volatility))
print('Multi-step MC price of an Barrier call option with 10% increased volatility is ' + str(callMC_Barrier_Knockin_Price_multi_step_increased_volatility)) 
print('Multi-step MC price of an Barrier put option with 10% increased volatility is ' + str(putMC_Barrier_Knockin_Price_multi_step_increased_volatility))


# get differences of prices from Black-Scholes Formula's less than 1 cent
np.random.seed(10)
callMC_European_Price_1_step_precise, putMC_European_Price_1_step_precise = \
  MC_european_price(S0, K, T, r, mu, sigma, numSteps = 1, numPaths = 10000000)
print('One-step MC price of an European call option with 10 Million Scenarios is ' + str(callMC_European_Price_1_step_precise))
print('One-step MC price of an European put option with 10 Million Scenarios is ' + str(putMC_European_Price_1_step_precise))