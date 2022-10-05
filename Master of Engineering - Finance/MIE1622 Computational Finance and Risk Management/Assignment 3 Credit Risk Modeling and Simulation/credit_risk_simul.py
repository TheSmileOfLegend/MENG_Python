# Import libraries
import numpy as np
import pandas as pd
import scipy
from scipy import special
from pathlib import Path
from scipy import sparse
import math
import scipy.stats as scs
import matplotlib.pyplot as plt

# set random seed so that the result is consistent every time run
np.random.seed(10)

Nout = 100000  # number of out-of-sample scenarios
Nin = 5000     # number of in-sample scenarios
Ns = 5         # number of idiosyncratic scenarios for each systemic

C = 8          # number of credit states

# Read and parse instrument data
instr_data = np.array(pd.read_csv('instrum_data.csv', header=None))
instr_id = instr_data[:, 0]     # ID
driver = instr_data[:, 1]       # credit driver
beta = instr_data[:, 2]         # beta (sensitivity to credit driver)
sigma = np.sqrt(1-np.square(beta))		# 1 - square of beta (sensitivity of idiosyncratic)
recov_rate = instr_data[:, 3]   # expected recovery rate
value = instr_data[:, 4]        # value
prob = instr_data[:, 5:(5 + C)] # credit-state migration probabilities (default to AAA)
exposure = instr_data[:, 5 + C:5 + 2 * C]  # credit-state migration exposures (default to AAA)
retn = instr_data[:, 5 + 2 * C] # market returns

K = instr_data.shape[0]         # number of CPs (counterparies)

# Read matrix of correlations for credit drivers
rho = np.array(pd.read_csv('credit_driver_corr.csv', sep='\t', header=None))

# Number of credit drivers
Ndriver = rho.shape[0]

# Cholesky decomp of rho (for generating correlated Normal random numbers)
sqrt_rho = np.linalg.cholesky(rho)

print('======= Credit Risk Model with Credit-State Migrations =======')
print('============== Monte Carlo Scenario Generation ===============')
print(' ')
print(' ')
print(' Number of out-of-sample Monte Carlo scenarios = ' + str(Nout))
print(' Number of in-sample Monte Carlo scenarios = ' + str(Nin))
print(' Number of counterparties = ' + str(K))
print(' ')

# Find credit-state for each counterparty
# 8 = AAA, 7 = AA, 6 = A, 5 = BBB, 4 = BB, 3 = B, 2 = CCC, 1 = default
CS = np.argmax(prob, axis=1) + 1

# Account for default recoveries
exposure[:, 0] = (1 - recov_rate) * exposure[:, 0]

# Compute credit-state boundaries
CS_Bdry = scipy.special.ndtri((np.cumsum(prob[:, 0:C - 1], 1)))

# -------- Insert your code here -------- #
filename_save_out = 'scen_out'

if Path(filename_save_out+'.npz').is_file():
    Losses_out = scipy.sparse.load_npz(filename_save_out + '.npz')
else:
    # Generating Scenarios
    # -------- Insert your code here -------- #
    Losses_out = np.zeros(K)
    losses_scenario = np.zeros(K)
    for s in range(1, Nout + 1):
        z_out_scenarios = np.random.normal(0,1,K)
        y_k_out_scenarios = np.random.normal(0,1,Ndriver)
        y_k_out_scenarios = np.dot(sqrt_rho,y_k_out_scenarios)

        # -------- Insert your code here -------- #
        y_j = np.array([y_k_out_scenarios[int(driver[j]) - 1] for j in range(K)])
        w_j = beta * y_j + sigma * z_out_scenarios

        for counterparty in range(K):
            losses_scenario[counterparty] = exposure[counterparty][7]
            for credit_state in range(C-1):
                if w_j[counterparty] < CS_Bdry[counterparty,credit_state]:
                    losses_scenario[counterparty] = exposure[counterparty,credit_state]
                    break
        Losses_out = np.vstack((Losses_out,losses_scenario))
    # Calculated out-of-sample losses (100000 x 100)
    # Losses_out (sparse matrix)
    Losses_out = np.delete(Losses_out, (0), axis=0)
    Losses_out = sparse.csc_matrix(Losses_out)
    sparse.save_npz(filename_save_out+'.npz',Losses_out)


# Normal approximation computed from out-of-sample scenarios
mu_l = np.mean(Losses_out, axis=0).reshape((K))
var_l = np.cov(Losses_out.toarray(), rowvar=False) # Losses_out as a sparse matrix

# Compute portfolio weights
portf_v = sum(value)  # portfolio value
w0 = []
w0.append(value / portf_v)   # asset weights (portfolio 1)
w0.append(np.ones((K)) / K)  # asset weights (portfolio 2)
x0 = []
x0.append((portf_v / value) * w0[0])  # asset units (portfolio 1)
x0.append((portf_v / value) * w0[1])  # asset units (portfolio 2)

# Quantile levels (99%, 99.9%)
alphas = np.array([0.99, 0.999])

VaRout = np.zeros((2, alphas.size))
VaRinN = np.zeros((2, alphas.size))
CVaRout = np.zeros((2, alphas.size))
CVaRinN = np.zeros((2, alphas.size))

loss_1y_plot = []
mean_out_plot = []
std_out_plot = []

for portN in range(2):
    loss_1y = np.dot(Losses_out.toarray(),np.array(x0[portN]))
    loss_1y = np.sort(loss_1y)

    # compute mean of portfolio
    mean_out = np.dot(mu_l,np.array(x0[portN]))
    
    # compute variance of portfolio
    var_out = 0
    for i in range(K):
    	for j in range(K):
    		var_out += var_l[i][j] * x0[portN][i] * x0[portN][j]
    
    # compute standard deviation of portfolio
    std_out = math.sqrt(var_out)

    # Compute VaR and CVaR
    for q in range(alphas.size):
        alf = alphas[q]
        # -------- Insert your code here -------- #
        VaRout[portN, q] = loss_1y[int(math.ceil(Nout * alf)) - 1]
        VaRinN[portN, q] = mean_out + scs.norm.ppf(alf) * std_out
        CVaRout[portN, q] = (1 / (Nout * (1 - alf))) * ((math.ceil(Nout * alf) - Nout * alf) * VaRout[portN, q] + sum(loss_1y[int(math.ceil(Nout * alf)):]))
        CVaRinN[portN, q] = mean_out + (scs.norm.pdf(scs.norm.ppf(alf)) / (1 - alf)) * std_out


    # store the loss data for plotting
    loss_1y_plot.append(loss_1y)
    mean_out_plot.append(mean_out)
    std_out_plot.append(std_out)

# set random seed so that the result is consistent every time run
# since this line of code is outside of trial looping, it will NOT cause each trial having same simulation result
np.random.seed(10)

# Perform 100 trials
N_trials = 100

VaRinMC1 = {}
VaRinMC2 = {}
VaRinN1 = {}
VaRinN2 = {}
CVaRinMC1 = {}
CVaRinMC2 = {}
CVaRinN1 = {}
CVaRinN2 = {}

for portN in range(2):
    for q in range(alphas.size):
        VaRinMC1[portN, q] = np.zeros(N_trials)
        VaRinMC2[portN, q] = np.zeros(N_trials)
        VaRinN1[portN, q] = np.zeros(N_trials)
        VaRinN2[portN, q] = np.zeros(N_trials)
        CVaRinMC1[portN, q] = np.zeros(N_trials)
        CVaRinMC2[portN, q] = np.zeros(N_trials)
        CVaRinN1[portN, q] = np.zeros(N_trials)
        CVaRinN2[portN, q] = np.zeros(N_trials)


for tr in range(1, N_trials + 1):
    # Monte Carlo approximation 1

    # -------- Insert your code here -------- #
    Losses_inMC1 = np.zeros(K)
    losses_scenario = np.zeros(K)

    for s in range(1, np.int(np.ceil(Nin / Ns) + 1)): # systemic scenarios
        # -------- Insert your code here -------- #
        y_k_MC1_scenarios = np.random.normal(0,1,Ndriver)
        y_k_MC1_scenarios = np.dot(sqrt_rho,y_k_MC1_scenarios)
        for si in range(1, Ns + 1): # idiosyncratic scenarios for each systemic
            # -------- Insert your code here -------- #
            z_MC1_scenarios = np.random.normal(0,1,K)
            y_j = np.array([y_k_MC1_scenarios[int(driver[j]) - 1] for j in range(K)])
            w_j = beta * y_j + sigma * z_MC1_scenarios
            for counterparty in range(K):
                losses_scenario[counterparty] = exposure[counterparty][7]
                for credit_state in range(C-1):
                    if w_j[counterparty] < CS_Bdry[counterparty,credit_state]:
                        losses_scenario[counterparty] = exposure[counterparty,credit_state]
                        break
            Losses_inMC1 = np.vstack((Losses_inMC1,losses_scenario))

    # Calculate losses for MC1 approximation (5000 x 100)
    # Losses_inMC1
    Losses_inMC1 = np.delete(Losses_inMC1, (0), axis=0)

    # Monte Carlo approximation 2

    # -------- Insert your code here -------- #
    Losses_inMC2 = np.zeros(K)

    for s in range(1, Nin + 1): # systemic scenarios (1 idiosyncratic scenario for each systemic)
        # -------- Insert your code here -------- #
        y_k_MC2_scenarios = np.random.normal(0,1,Ndriver)
        y_k_MC2_scenarios = np.dot(sqrt_rho,y_k_MC2_scenarios)
        z_MC2_scenarios = np.random.normal(0,1,K)
        y_j = np.array([y_k_MC2_scenarios[int(driver[j]) - 1] for j in range(K)])
        w_j = beta * y_j + sigma * z_MC2_scenarios
        for counterparty in range(K):
            losses_scenario[counterparty] = exposure[counterparty][7]
            for credit_state in range(C-1):
                if w_j[counterparty] < CS_Bdry[counterparty,credit_state]:
                    losses_scenario[counterparty] = exposure[counterparty,credit_state]
                    break
        Losses_inMC2 = np.vstack((Losses_inMC2,losses_scenario))
    # Calculated losses for MC2 approximation (5000 x 100)
    # Losses_inMC2
    Losses_inMC2 = np.delete(Losses_inMC2, (0), axis=0)

    # Compute VaR and CVaR

    for portN in range(2):
        for q in range(alphas.size):
            alf = alphas[q]
            # -------- Insert your code here -------- #
            # Compute portfolio loss
            portf_loss_inMC1 = np.dot(Losses_inMC1,np.array(x0[portN]))
            portf_loss_inMC2 = np.dot(Losses_inMC2,np.array(x0[portN]))
            portf_loss_inMC1 = np.sort(portf_loss_inMC1)
            portf_loss_inMC2 = np.sort(portf_loss_inMC2)
            mu_MC1 = np.mean(Losses_inMC1, axis=0).reshape((K))
            var_MC1 = np.cov(Losses_inMC1, rowvar=False)
            mu_MC2 = np.mean(Losses_inMC2, axis=0).reshape((K))
            var_MC2 = np.cov(Losses_inMC2, rowvar=False)
            # Compute portfolio mean loss mu_p_MC1 and portfolio standard deviation of losses sigma_p_MC1
            # Compute portfolio mean loss mu_p_MC2 and portfolio standard deviation of losses sigma_p_MC2
            # Compute VaR and CVaR for the current trial
            mu_p_MC1 = np.dot(mu_MC1,np.array(x0[portN]))

            # compute variance of portfolio for MC1
            var_p_MC1 = 0
            for i in range(K):
            	for j in range(K):
            		var_p_MC1 += var_MC1[i][j] * x0[portN][i] * x0[portN][j]
            
            sigma_p_MC1 = math.sqrt(var_p_MC1)
           
            mu_p_MC2 = np.dot(mu_MC2,np.array(x0[portN]))

            # compute variance of portfolio for MC2
            var_p_MC2 = 0
            for i in range(K):
            	for j in range(K):
            		var_p_MC2 += var_MC2[i][j] * x0[portN][i] * x0[portN][j]
            
            sigma_p_MC2 = math.sqrt(var_p_MC2)

            VaRinMC1[portN, q][tr - 1] = portf_loss_inMC1[int(math.ceil(Nin * alf)) - 1]
            VaRinMC2[portN, q][tr - 1] = portf_loss_inMC2[int(math.ceil(Nin * alf)) - 1]
            VaRinN1[portN, q][tr - 1] =  mu_p_MC1 + scs.norm.ppf(alf) * sigma_p_MC1
            VaRinN2[portN, q][tr - 1] =  mu_p_MC2 + scs.norm.ppf(alf) * sigma_p_MC2
            CVaRinMC1[portN, q][tr - 1] = (1 / (Nin * (1 - alf))) * ((math.ceil(Nin * alf) - Nin * alf) * VaRinMC1[portN, q][tr - 1] + sum(portf_loss_inMC1[int(math.ceil(Nin * alf)):]))
            CVaRinMC2[portN, q][tr - 1] = (1 / (Nin * (1 - alf))) * ((math.ceil(Nin * alf) - Nin * alf) * VaRinMC2[portN, q][tr - 1] + sum(portf_loss_inMC2[int(math.ceil(Nin * alf)):]))
            CVaRinN1[portN, q][tr - 1] =  mu_p_MC1 + (scs.norm.pdf(scs.norm.ppf(alf)) / (1 - alf)) * sigma_p_MC1
            CVaRinN2[portN, q][tr - 1] =  mu_p_MC2 + (scs.norm.pdf(scs.norm.ppf(alf)) / (1 - alf)) * sigma_p_MC2

# Display VaR and CVaR

for portN in range(2):
    print('\nPortfolio {}:\n'.format(portN + 1))
    for q in range(alphas.size):
        alf = alphas[q]
        print('Out-of-sample: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, VaRout[portN, q], 100 * alf, CVaRout[portN, q]))
        print('In-sample MC1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, np.mean(VaRinMC1[portN, q]), 100 * alf, np.mean(CVaRinMC1[portN, q])))
        print('In-sample MC2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, np.mean(VaRinMC2[portN, q]), 100 * alf, np.mean(CVaRinMC2[portN, q])))
        print('In-sample No: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, VaRinN[portN, q], 100 * alf, CVaRinN[portN, q]))
        print('In-sample N1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, np.mean(VaRinN1[portN, q]), 100 * alf, np.mean(CVaRinN1[portN, q])))
        print('In-sample N2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n' % (
        100 * alf, np.mean(VaRinN2[portN, q]), 100 * alf, np.mean(CVaRinN2[portN, q])))

# Plot results
# Figure 1: loss distribution of out-of-sample (portfolio 1) with VaR
frequencyCounts, binLocations, patches = plt.hist(loss_1y_plot[0], 100)
normf_out = (1/(std_out_plot[0] * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - mean_out_plot[0].tolist()[0][0]) / std_out_plot[0]) ** 2)
normf_out = normf_out * sum(frequencyCounts) / sum(normf_out)
plt.plot(binLocations, normf_out, color = 'r', linewidth =3.0)
plt.plot([VaRinN[0, 0], VaRinN[0,0]], [0, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='-')
plt.plot([VaRout[0, 0], VaRout[0,0]], [0, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='-')
plt.text(0.95 * VaRinN[0, 0], max(frequencyCounts) / 1.9, '99%VaRn_out',rotation = 'vertical', color='r')
plt.text(0.95 * VaRout[0, 0], max(frequencyCounts) / 1.9, '99%VaR_out',rotation = 'vertical', color='b')
plt.plot([VaRinN[0, 1], VaRinN[0,1]], [0, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.plot([VaRout[0, 1], VaRout[0,1]], [0, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.text(0.95 * VaRinN[0, 1], max(frequencyCounts) / 1.9, '99.9%VaRn_out',rotation = 'vertical', color='r')
plt.text(0.98 * VaRout[0, 1], max(frequencyCounts) / 1.9, '99.9%VaR_out',rotation = 'vertical', color='b')

# MC1 mean Var and CVar
plt.plot([np.mean(VaRinMC1[0, 0]), np.mean(VaRinMC1[0, 0])], [0, max(frequencyCounts) / 2], color='lime', linewidth=1.3, linestyle='-')
plt.plot([np.mean(VaRinMC1[0, 1]), np.mean(VaRinMC1[0, 1])], [0, max(frequencyCounts)/2], color='lime', linewidth=1.3, linestyle='--',dashes=(5, 3))

# MC2 mean Var and CVar
plt.plot([np.mean(VaRinMC2[0, 0]), np.mean(VaRinMC2[0, 0])], [0, max(frequencyCounts) / 2], color='magenta', linewidth=1.3, linestyle='-')
plt.plot([np.mean(VaRinMC2[0, 1]), np.mean(VaRinMC2[0, 1])], [0, max(frequencyCounts)/2], color='magenta', linewidth=1.3, linestyle='--',dashes=(5, 3))

plt.legend(['out normal dist.','99%VaRn_out','99%VaR_out','99.9%VaRn_out','99.9%VaR_out','99%VaR_MC1_mean','99.9%VaR_MC1_mean','99%VaR_MC2_mean','99.9%VaR_MC2_mean','out true dist.'],bbox_to_anchor=(1, 1),loc='upper left')
plt.xlabel('1-Year Loss in $ Value on Portfolio 1')
plt.ylabel('Frequency')
plt.title('Loss Distribution of Portfolio 1 with VaR')
plt.draw()

# Figure 2: loss distribution of out-of-sample (portfolio 2) with VaR
frequencyCounts, binLocations, patches = plt.hist(loss_1y_plot[1], 100)
normf_out = (1/(std_out_plot[1] * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - mean_out_plot[1].tolist()[0][0]) / std_out_plot[1]) ** 2)
normf_out = normf_out * sum(frequencyCounts) / sum(normf_out)
plt.plot(binLocations, normf_out, color = 'r', linewidth =3.0)
plt.plot([VaRinN[1, 0], VaRinN[1,0]], [1, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='-')
plt.plot([VaRout[1, 0], VaRout[1,0]], [1, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='-')
plt.text(0.95 * VaRinN[1, 0], max(frequencyCounts) / 1.9, '99%VaRn_out',rotation = 'vertical', color='r')
plt.text(0.95 * VaRout[1, 0], max(frequencyCounts) / 1.9, '99%VaR_out',rotation = 'vertical', color='b')
plt.plot([VaRinN[1, 1], VaRinN[1,1]], [0, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.plot([VaRout[1, 1], VaRout[1,1]], [0, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.text(0.95 * VaRinN[1, 1], max(frequencyCounts) / 1.9, '99.9%VaRn_out',rotation = 'vertical', color='r')
plt.text(0.98 * VaRout[1, 1], max(frequencyCounts) / 1.9, '99.9%VaR_out',rotation = 'vertical', color='b')

# MC1 mean Var and CVar
plt.plot([np.mean(VaRinMC1[1, 0]), np.mean(VaRinMC1[1, 0])], [1, max(frequencyCounts) / 2], color='lime', linewidth=1.3, linestyle='-')
plt.plot([np.mean(VaRinMC1[1, 1]), np.mean(VaRinMC1[1, 1])], [1, max(frequencyCounts)/2], color='lime', linewidth=1.3, linestyle='--',dashes=(5, 3))

# MC2 mean Var and CVar
plt.plot([np.mean(VaRinMC2[1, 0]), np.mean(VaRinMC2[1, 0])], [0, max(frequencyCounts) / 2], color='magenta', linewidth=1.3, linestyle='-')
plt.plot([np.mean(VaRinMC2[1, 1]), np.mean(VaRinMC2[1, 1])], [0, max(frequencyCounts)/2], color='magenta', linewidth=1.3, linestyle='--',dashes=(5, 3))

plt.legend(['out normal dist.','99%VaRn_out','99%VaR_out','99.9%VaRn_out','99.9%VaR_out','99%VaR_MC1_mean','99.9%VaR_MC1_mean','99%VaR_MC2_mean','99.9%VaR_MC2_mean','out true dist.'],bbox_to_anchor=(1, 1),loc='upper left')
plt.xlabel('1-Year Loss in $ Value on Portfolio 2')
plt.ylabel('Frequency')
plt.title('Loss Distribution of Portfolio 2 with VaR')
plt.draw()

# Figure 3: loss distribution of out-of-sample (portfolio 1) with CVaR
frequencyCounts, binLocations, patches = plt.hist(loss_1y_plot[0], 100)
normf_out = (1/(std_out_plot[0] * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - mean_out_plot[0].tolist()[0][0]) / std_out_plot[0]) ** 2)
normf_out = normf_out * sum(frequencyCounts) / sum(normf_out)
plt.plot(binLocations, normf_out, color = 'r', linewidth =3.0)
plt.plot([CVaRinN[0, 0], CVaRinN[0,0]], [0, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='-')
plt.plot([CVaRout[0, 0], CVaRout[0,0]], [0, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='-')
plt.text(0.95 * CVaRinN[0, 0], max(frequencyCounts) / 1.9, '99%CVaRn_out',rotation = 'vertical', color='r')
plt.text(0.95 * CVaRout[0, 0], max(frequencyCounts) / 1.9, '99%CVaR_out',rotation = 'vertical', color='b')
plt.plot([CVaRinN[0, 1], CVaRinN[0,1]], [0, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.plot([CVaRout[0, 1], CVaRout[0,1]], [0, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.text(0.95 * CVaRinN[0, 1], max(frequencyCounts) / 1.9, '99.9%CVaRn_out',rotation = 'vertical', color='r')
plt.text(0.98 * CVaRout[0, 1], max(frequencyCounts) / 1.9, '99.9%CVaR_out',rotation = 'vertical', color='b')

# MC1 mean Var and CVar
plt.plot([np.mean(CVaRinMC1[0, 0]), np.mean(CVaRinMC1[0, 0])], [0, max(frequencyCounts) / 2], color='lime', linewidth=1.3, linestyle='-')
plt.plot([np.mean(CVaRinMC1[0, 1]), np.mean(CVaRinMC1[0, 1])], [0, max(frequencyCounts)/2], color='lime', linewidth=1.3, linestyle='--',dashes=(5, 3))

# MC2 mean Var and CVar
plt.plot([np.mean(CVaRinMC2[0, 0]), np.mean(CVaRinMC2[0, 0])], [0, max(frequencyCounts) / 2], color='magenta', linewidth=1.3, linestyle='-')
plt.plot([np.mean(CVaRinMC2[0, 1]), np.mean(CVaRinMC2[0, 1])], [0, max(frequencyCounts)/2], color='magenta', linewidth=1.3, linestyle='--',dashes=(5, 3))

plt.legend(['out normal dist.','99%CVaRn_out','99%CVaR_out','99.9%CVaRn_out','99.9%CVaR_out','99%CVaR_MC1_mean','99.9%CVaR_MC1_mean','99%CVaR_MC2_mean','99.9%CVaR_MC2_mean','out true dist.'],bbox_to_anchor=(1, 1),loc='upper left')
plt.xlabel('1-Year Loss in $ Value on Portfolio 1')
plt.ylabel('Frequency')
plt.title('Loss Distribution of Portfolio 1 with CVaR')
plt.draw()

# Figure 4: loss distribution of out-of-sample (portfolio 2) with CVaR
frequencyCounts, binLocations, patches = plt.hist(loss_1y_plot[1], 100)
normf_out = (1/(std_out_plot[1] * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - mean_out_plot[1].tolist()[0][0]) / std_out_plot[1]) ** 2)
normf_out = normf_out * sum(frequencyCounts) / sum(normf_out)
plt.plot(binLocations, normf_out, color = 'r', linewidth =3.0)
plt.plot([CVaRinN[1, 0], CVaRinN[1,0]], [1, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='-')
plt.plot([CVaRout[1, 0], CVaRout[1,0]], [1, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='-')
plt.text(0.95 * CVaRinN[1, 0], max(frequencyCounts) / 1.9, '99%CVaRn_out',rotation = 'vertical', color='r')
plt.text(0.95 * CVaRout[1, 0], max(frequencyCounts) / 1.9, '99%CVaR_out',rotation = 'vertical', color='b')
plt.plot([CVaRinN[1, 1], CVaRinN[1,1]], [0, max(frequencyCounts) / 2], color='r', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.plot([CVaRout[1, 1], CVaRout[1,1]], [0, max(frequencyCounts)/2], color='b', linewidth=1.3, linestyle='--',dashes=(5, 3))
plt.text(0.95 * CVaRinN[1, 1], max(frequencyCounts) / 1.9, '99.9%CVaRn_out',rotation = 'vertical', color='r')
plt.text(0.98 * CVaRout[1, 1], max(frequencyCounts) / 1.9, '99.9%CVaR_out',rotation = 'vertical', color='b')

# MC1 mean Var and CVar
plt.plot([np.mean(CVaRinMC1[1, 0]), np.mean(CVaRinMC1[1, 0])], [1, max(frequencyCounts) / 2], color='lime', linewidth=1.3, linestyle='-')
plt.plot([np.mean(CVaRinMC1[1, 1]), np.mean(CVaRinMC1[1, 1])], [1, max(frequencyCounts)/2], color='lime', linewidth=1.3, linestyle='--',dashes=(5, 3))

# MC2 mean Var and CVar
plt.plot([np.mean(CVaRinMC2[1, 0]), np.mean(CVaRinMC2[1, 0])], [0, max(frequencyCounts) / 2], color='magenta', linewidth=1.3, linestyle='-')
plt.plot([np.mean(CVaRinMC2[1, 1]), np.mean(CVaRinMC2[1, 1])], [0, max(frequencyCounts)/2], color='magenta', linewidth=1.3, linestyle='--',dashes=(5, 3))

plt.legend(['out normal dist.','99%CVaRn_out','99%CVaR_out','99.9%CVaRn_out','99.9%CVaR_out','99%CVaR_MC1_mean','99.9%CVaR_MC1_mean','99%CVaR_MC2_mean','99.9%CVaR_MC2_mean','out true dist.'],bbox_to_anchor=(1, 1),loc='upper left')
plt.xlabel('1-Year Loss in $ Value on Portfolio 2')
plt.ylabel('Frequency')
plt.title('Loss Distribution of Portfolio 2 with CVaR')
plt.draw()

# Figure 5: standard deviation comparison for portfolio 1
port_1_std = {}
port_1_std['99% VaR MC1'] = np.std(VaRinMC1[0, 0])
port_1_std['99% VaR MC2'] = np.std(VaRinMC2[0, 0])
port_1_std['99% VaR N1'] = np.std(VaRinN1[0, 0])
port_1_std['99% VaR N2'] = np.std(VaRinN2[0, 0])
port_1_std['99% CVaR MC1'] = np.std(CVaRinMC1[0, 0])
port_1_std['99% CVaR MC2'] = np.std(CVaRinMC2[0, 0])
port_1_std['99% CVaR N1'] = np.std(CVaRinN1[0, 0])
port_1_std['99% CVaR N2'] = np.std(CVaRinN2[0, 0])
port_1_std['99.9% VaR MC1'] = np.std(VaRinMC1[0, 1])
port_1_std['99.9% VaR MC2'] = np.std(VaRinMC2[0, 1])
port_1_std['99.9% VaR N1'] = np.std(VaRinN1[0, 1])
port_1_std['99.9% VaR N2'] = np.std(VaRinN2[0, 1])
port_1_std['99.9% CVaR MC1'] = np.std(CVaRinMC1[0, 1])
port_1_std['99.9% CVaR MC2'] = np.std(CVaRinMC2[0, 1])
port_1_std['99.9% CVaR N1'] = np.std(CVaRinN1[0, 1])
port_1_std['99.9% CVaR N2'] = np.std(CVaRinN2[0, 1])
port_1_std = dict(sorted(port_1_std.items(), key=lambda x: x[1]))
keys = port_1_std.keys()
values = port_1_std.values()
plt.title('VaR and CVaR Standard Deviation of 100 Trials Comparison for Portfolio 1',y=1.08)
plt.bar(keys, values)
plt.xticks(rotation='vertical')
plt.ylabel('Standard Deviation of Trial($)')
plt.xlabel('Simulation Setting and Parameter')
plt.draw()

# Figure 6: standard deviation comparison for portfolio 2
port_2_std = {}
port_2_std['99% VaR MC1'] = np.std(VaRinMC1[1, 0])
port_2_std['99% VaR MC2'] = np.std(VaRinMC2[1, 0])
port_2_std['99% VaR N1'] = np.std(VaRinN1[1, 0])
port_2_std['99% VaR N2'] = np.std(VaRinN2[1, 0])
port_2_std['99% CVaR MC1'] = np.std(CVaRinMC1[1, 0])
port_2_std['99% CVaR MC2'] = np.std(CVaRinMC2[1, 0])
port_2_std['99% CVaR N1'] = np.std(CVaRinN1[1, 0])
port_2_std['99% CVaR N2'] = np.std(CVaRinN2[1, 0])
port_2_std['99.9% VaR MC1'] = np.std(VaRinMC1[1, 1])
port_2_std['99.9% VaR MC2'] = np.std(VaRinMC2[1, 1])
port_2_std['99.9% VaR N1'] = np.std(VaRinN1[1, 1])
port_2_std['99.9% VaR N2'] = np.std(VaRinN2[1, 1])
port_2_std['99.9% CVaR MC1'] = np.std(CVaRinMC1[1, 1])
port_2_std['99.9% CVaR MC2'] = np.std(CVaRinMC2[1, 1])
port_2_std['99.9% CVaR N1'] = np.std(CVaRinN1[1, 1])
port_2_std['99.9% CVaR N2'] = np.std(CVaRinN2[1, 1])
port_2_std = dict(sorted(port_2_std.items(), key=lambda x: x[1]))
keys = port_2_std.keys()
values = port_2_std.values()
plt.title('VaR and CVaR Standard Deviation of 100 Trials Comparison for Portfolio 2',y=1.08)
plt.bar(keys, values)
plt.xticks(rotation='vertical')
plt.ylabel('Standard Deviation of Trial($)')
plt.xlabel('Simulation Setting and Parameter')
plt.draw()