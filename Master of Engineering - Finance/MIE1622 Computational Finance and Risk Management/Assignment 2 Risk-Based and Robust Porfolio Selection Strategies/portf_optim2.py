# Import libraries
import pandas as pd
import numpy as np
import math
import cplex
import matplotlib.pyplot as plt
from datetime import date
import ipopt
import copy
class erc(object):
    def __init__(self):
        pass

    def objective(self, x):
        # The callback for calculating the objective
        y = x * np.dot(Q, x)
        fval = 0
        for i in range(N):
            for j in range(i,N):
                xij = y[i] - y[j]
                fval = fval + xij*xij
        fval = 2*fval
        return fval

    def gradient(self, x):
        # The callback for calculating the gradient
        grad = np.zeros(N)
        # Insert your gradient computations here
        # You can use finite differences to check the gradient
        y = x * np.dot(Q,x)
        for i in range(N):
            for j in range(i + 1, N):
                if i != j:
                    for grad_i in range(N):
                        if i == grad_i:
                            grad[grad_i] += 2 * (y[i] - y[j]) * (y[i] + x[i] * Q[i,i] - x[j] * Q[j,i])
                        elif j == grad_i:
                            grad[grad_i] += 2 * (y[j] - y[i]) * (y[j] + x[j] * Q[j,j] - x[i] * Q[i,j])
                        else:
                            grad[grad_i] += 2 * (y[i] - y[j]) * (x[i] * Q[i,grad_i] - x[j] * Q[j,grad_i])
        grad = grad * 2
        return grad

    def constraints(self, x):
        # The callback for calculating the constraints
        return [1.0] * N
    
    def jacobian(self, x):
        # The callback for calculating the Jacobian
        return np.array([[1.0] * N])

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        # Example for the use of the intermediate callback. uncomment next line for print
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        return

# Complete the following functions
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
    x_optimal = x_init
    cash_optimal = cash_init
    return x_optimal, cash_optimal

def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    # compute weights of assets in portfolio
    weights = np.array([1/N] * N)

    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost
    return x_optimal, cash_optimal

def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    # optimization for min variance to derive optimal weights of assets in portfolio
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0] * N
    ub = [cplex.infinity] * N
    lb = [0] * N
    cols = [[[0],[1]]] * N
    cpx.linear_constraints.add(rhs = [1.0], senses = "E")
    var_names = ["w%s" % i for i in range(1,N + 1)] 
    cpx.variables.add(obj = c, lb = lb, ub = ub, columns = cols, names = var_names)
    sparse_indices = [i for i in range(N)]
    qmat = [[sparse_indices, i] for i in (2 * Q).tolist()]
    cpx.objective.set_quadratic(qmat)
    alg = cpx.parameters.lpmethod.values
    cpx.parameters.qpmethod.set(alg.concurrent)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.solve()
    weights = np.array(cpx.solution.get_values())
  
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost

    return x_optimal, cash_optimal, weights

def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    # optimization for max Sharpe ratio to derive optimal weights of assets in portfolio
    rf_daily = r_rf/252
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0] * (N + 1)
    ub = [cplex.infinity] * (N + 1)
    lb = [0] * (N + 1)
    cols = [[[0,1],[mu[i] - rf_daily,1]] for i in range(N)]
    cols.append([[0,1],[0,-1]])
    cpx.linear_constraints.add(rhs = [1.0,0], senses = "EE")
    var_names = ["y%s" % i for i in range(1,N + 2)] 
    cpx.variables.add(obj = c, lb=lb, ub = ub, columns = cols, names = var_names)
    sparse_indices = [i for i in range(N)]
    qmat = [[sparse_indices, i] for i in (2*Q).tolist()]
    qmat.append([[],[]])
    cpx.objective.set_quadratic(qmat)
    alg = cpx.parameters.lpmethod.values
    cpx.parameters.qpmethod.set(alg.concurrent)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.solve()
    ys = np.array(cpx.solution.get_values())
    weights = ys[:N] / ys[N]

    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost

    return x_optimal, cash_optimal

def strat_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices):
    # optimization for ERC to derive weights of assets in portfolio
    # Use "1/n portfolio" w0 as initial portfolio for starting IPOPT optimization
    lb = [0.0] * N  # lower bounds on variables
    ub = [1.0] * N  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints
    w0 = [1.0/N] * N

    # Define IPOPT problem
    nlp = ipopt.problem(n=len(w0), m=len(cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu)
     
    # Set the IPOPT options
    nlp.addOption('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8'))
    nlp.addOption('hessian_approximation'.encode('utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.addOption('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.addOption('tol'.encode('utf-8'), 1e-10)

    # Solve the problem
    weights, info = nlp.solve(w0)
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost

    return x_optimal, cash_optimal, weights

def strat_lever_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices, erc_weights, rf_asset_value_init):
    # assumption: no transaction cost associated with shorting with risk-free interest
    portf_value_init = x_init.dot(cur_prices) + cash_init - rf_asset_value_init

    # inherit weights from erc portfolio
    weights = erc_weights

    # compute the portfolio before reborrowing
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init - rf_asset_value_init 
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal_1 = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal_1 - x_init).dot(cur_prices) * 0.005
    cash_optimal_1 = portf_value_init - x_optimal_1.dot(cur_prices) - trans_cost

    # money to be borrowed this period
    rf_asset_value_cur = x_optimal_1.dot(cur_prices) + cash_optimal_1

    # compute the portfolio after reborrowing
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_optimal_1.dot(cur_prices) + cash_optimal_1 + rf_asset_value_cur
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_optimal_1).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal_2 = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal_2 - x_optimal_1).dot(cur_prices) * 0.005
    cash_optimal_2 = portf_value_init - x_optimal_2.dot(cur_prices) - trans_cost
    
    return x_optimal_2, cash_optimal_2, rf_asset_value_cur

def strat_robust_optim(x_init, cash_init, mu, Q, cur_prices, w_minVar):
    # Define initial portfolio ("equally weighted" or "1/n portfolio")
    w0 = [1.0/N] * N
    
    # Sanity check
    Sum_w = sum(w0)
    
    # 1/n portfolio return
    ret_init = np.dot(mu, w0)
    
    # 1/n portfolio variance
    var_init = np.dot(w0, np.dot(Q, w0))

    # Required portfolio robustness
    var_matr = np.diag(np.diag(Q))

    # Target portfolio return estimation error is return estimation error of 1/n portfolio
    rob_init = np.dot(w0, np.dot(var_matr, w0)) # return estimation error of initial portfolio
    rob_bnd  = rob_init # target return estimation error
    
    var_minVar = np.dot(w_minVar, np.dot(Q, w_minVar))
    ret_minVar = np.dot(mu, w_minVar)
    rob_minVar = np.dot(w_minVar, np.dot(var_matr, w_minVar))
    
    # Target portfolio return is 20% more (30% has no solution) than return of minimum variance portfolio
    Portf_Retn = ret_minVar * 1.2
    
    Qq_rMV = var_matr
    Qq_rMVs = np.sqrt(Qq_rMV)
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c  = [0.0] * N
    lb = [0.0] * N
    ub = [1.0] * N
    A = []
    for k in range(N):
        A.append([[0,1],[1.0,mu[k]]])
   
    var_names = ["w_%s" % i for i in range(1,N+1)]
    cpx.linear_constraints.add(rhs=[1.0,Portf_Retn], senses="EG")
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(N)), list(2*Q[k,:])] for k in range(N)]
    cpx.objective.set_quadratic(Qmat)
    Qcon = cplex.SparseTriple(ind1=var_names, ind2=range(N), val=np.diag(var_matr))
    cpx.quadratic_constraints.add(rhs=rob_bnd, quad_expr=Qcon, name="Qc")
    cpx.parameters.threads.set(4)
    cpx.parameters.timelimit.set(60)
    cpx.parameters.barrier.qcpconvergetol.set(1e-12)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.solve()
    w_rMV = cpx.solution.get_values()
    card_rMV = np.count_nonzero(w_rMV)
    ret_rMV  = np.dot(mu, w_rMV)
    var_rMV = np.dot(w_rMV, np.dot(Q, w_rMV))
    rob_rMV = np.dot(w_rMV, np.dot(var_matr, w_rMV))
    # Round near-zero portfolio weights
    w_rMV = np.array(w_rMV)
    w_rMV_nonrnd = copy.deepcopy(w_rMV)
    w_rMV[w_rMV<1e-6] = 0
    w_rMV = w_rMV / np.sum(w_rMV)

    weights = w_rMV

    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost
    return x_optimal, cash_optimal

# Input file
input_file_prices = 'Daily_closing_prices.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)

# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])
# Find the number of trading days in Nov-Dec 2018 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2018)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)

# Remove datapoints for year 2018
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]

# Initial positions in the portfolio
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])

# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))

# Initial portfolio weights
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2019-2020 is 2.5%
r_rf = 0.025
# Annual risk-free rate for years 2008-2009 is 4.5%
r_rf2008_2009 = 0.045

# Create a list recording prices at beginning of periods
period_prices = []

# Create a list recording maximum drawdown in periods
max_drawdown = []

# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe', 'strat_equal_risk_contr', 'strat_lever_equal_risk_contr', 'strat_robust_optim']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Mininum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio', 'Equal Risk Contributions Portfolio', 'Leveraged Equal Risk Contributions Portfolio', 'Robust Optimization Portfolio']
# N_strat = 7  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe, strat_equal_risk_contr, strat_lever_equal_risk_contr, strat_robust_optim]

portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
rf_asset_value = 0
for period in range(1, N_periods+1):
    # Compute current year and month, first and last day of the period
    if dates_array[0, 0] == 19:
        cur_year  = 19 + math.floor(period/7)
    else:
        cur_year  = 2019 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
    print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   
    # Prices for the current day
    cur_prices = data_prices[day_ind_start,:]
    period_prices.append(cur_prices)

    # Create a list recording maximum drawdown in each period
    max_drawdown_period = []

    # Execute portfolio selection strategies
    for strategy  in range(N_strat):

        # Get current portfolio positions
        if period == 1:
          curr_positions = init_positions
          curr_cash = 0
          portf_value[strategy] = np.zeros((N_days, 1))
        else:
          curr_positions = x[strategy, period-2]
          curr_cash = cash[strategy, period-2]

        # Compute strategy
        # additional set of weights received from erc portfolio function, then pass the weights to leveraged erc, receiving additional record of risk-free asset amount
        if strategy == 4: 
          x[strategy, period-1], cash[strategy, period-1], erc_weights = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        elif strategy == 5:
          x[strategy, period-1], cash[strategy, period-1], rf_asset_value = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices, erc_weights, rf_asset_value)
        elif strategy == 2:
          x[strategy, period-1], cash[strategy, period-1], w_minVar = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        elif strategy == 6:
          x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices, w_minVar)
        else:
          x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
        # Check that cash account is >= 0
        # Check that we can buy new portfolio subject to transaction costs

        ###################### Insert your code here ############################
        position_pre = curr_positions
        position_post = x[strategy, period - 1]

        # balance of shares sold deducting shares purchased 
        balance = (position_pre - position_post).dot(cur_prices)
    
        # transaction cost
        trans_cost = abs(position_pre - position_post).dot(cur_prices) * 0.005

        # identifying if cash account is nonnegative
        flag_cash_nonneg =  cash[strategy, period - 1] >= 0

        # identifying if the new portfolio and the associated transaction cost can be covered fully by the budget
        if strategy == 5:
          flag_enough_budget = (curr_cash + rf_asset_value) >= trans_cost + ((position_pre + position_post)/2).dot(cur_prices) * 0.005

        else:
          flag_enough_budget = (curr_cash + balance) >= trans_cost

        # verification of portfolio feasibility and print a message if the portfolio is infeasible
        if not(flag_cash_nonneg) or not(flag_enough_budget):
          print()
          print("Strategy is infeasible.")
          print()

        # Compute portfolio value
        p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
        portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))

        #subtract money borrowed and its interest generated from portfolio value
        if strategy ==5:
          for i in range(day_ind_start, day_ind_end+1):
            portf_value[5][i] -= rf_asset_value * (1 + r_rf / 6 * (i - day_ind_start) / (day_ind_end - day_ind_start))
        
        print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
              portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))
        
        # include risk-free interest for 2 months (one period)
        rf_asset_value *= (1 + r_rf / 6)


        max_drawdown_period.append((max(portf_value[strategy][day_ind_start:day_ind_end]) - min(portf_value[strategy][day_ind_start:day_ind_end]))/max(portf_value[strategy][day_ind_start:day_ind_end])[0])

    # Compute expected returns and covariances for the next period
    cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
    mu = np.mean(cur_returns, axis = 0)
    Q = np.cov(cur_returns.T)
    max_drawdown.append(max_drawdown_period)
# Plot results
###################### Insert your code here ############################

# Plot of portfolio value for different strategies
df_daily_portf_value = pd.DataFrame(np.array([i.flatten() for i in portf_value]).T, index = pd.DatetimeIndex(dates), columns = strategy_names)
df_daily_portf_value.plot()
start_date = date.fromisoformat('2019-01-02')
end_date = date.fromisoformat('2020-12-31')
plt.xlim([start_date, end_date])
plt.title('Portfolio Value for Different Strategies from 2019 to 2020', fontweight="bold", pad = 20, fontsize="12")
plt.xlabel('Date', fontweight="bold", fontsize="12")
plt.ylabel('Portfolio Value', fontweight="bold", fontsize="12")
plt.show()

# Plot of dynamic changes (in terms of weights of value) in portfolio allocation for strategy 7
# get name list of stocks
stock_list = df.columns.tolist()[1:]
weights_data = []
for strategy  in range(6,7):
    stock_data = {}
    for stock in range(N):
        weights_period = []
        for period in range(N_periods):
                weights_period.append(x[strategy, period][stock] * period_prices[period][stock] / (x[strategy, period].dot(period_prices[period]) + cash[strategy, period]))
        stock_data[stock_list[stock]] = weights_period
    weights_data.append(stock_data)

periods = [i for i in range(1, N_periods + 1)]

color_map = ["#9ACD32", "#00FF7F", "#EE82EE", "#40E0D0", "#FF6347", "#4682B4", "#6A5ACD","#000080", "#FF0000", "#FFFFE0", "#000000", "#696969", "#800080","#FFFF00","#B8860B","#E0FFFF","#008000","#FF69B4", "#0000CD", "#A52A2A"]
for i in range(1):
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 12])
    ax.stackplot(periods, weights_data[i].values(),labels=weights_data[i].keys(),colors = color_map)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.set_title('Portfolio Composition vs. Periods for Strategy 7', fontweight="bold", fontsize="12")
    ax.set_xlabel('Period', fontweight="bold", fontsize="12")
    ax.set_ylabel('Value Weights (Accumulated)', fontweight="bold", fontsize="12")
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.show()

# Plot of dynamic changes (in terms of weights of position) in portfolio allocation for strategy 7
# get name list of stocks
stock_list = df.columns.tolist()[1:]
weights_data = []
for strategy  in range(6,7):
    stock_data = {}
    for stock in range(N):
        weights_period = []
        for period in range(N_periods):
                weights_period.append(x[strategy, period][stock] / sum(x[strategy, period]))
        stock_data[stock_list[stock]] = weights_period
    weights_data.append(stock_data)

periods = [i for i in range(1, N_periods + 1)]

color_map = ["#9ACD32", "#00FF7F", "#EE82EE", "#40E0D0", "#FF6347", "#4682B4", "#6A5ACD","#000080", "#FF0000", "#FFFFE0", "#000000", "#696969", "#800080","#FFFF00","#B8860B","#E0FFFF","#008000","#FF69B4", "#0000CD", "#A52A2A"]
for i in range(1):
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 12])
    ax.stackplot(periods, weights_data[i].values(),labels=weights_data[i].keys(),colors = color_map)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.set_title('Portfolio Composition vs. Periods for Strategy 7', fontweight="bold", fontsize="12")
    ax.set_xlabel('Period', fontweight="bold", fontsize="12")
    ax.set_ylabel('Position Weights (Accumulated)', fontweight="bold", fontsize="12")
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.show()



# Plot of maximum drawdown for different strategies
df_max_drawdown = pd.DataFrame(np.array([i.flatten() for i in np.array(max_drawdown)]), index = [i for i in range(1,13)], columns = strategy_names)
df_max_drawdown.plot()
plt.title('Maximum Drawdown (absolute value) for Different Strategies from 2019 to 2020', fontweight="bold", pad = 20, fontsize="12")
plt.xlabel('Period', fontweight="bold", fontsize="12")
plt.ylabel('Maximum Drawdown', fontweight="bold", fontsize="12")
plt.show()

# Repeated Coding for 2008 - 2009
#**************************************************************************************************************************************************************************8
# Import libraries
import pandas as pd
import numpy as np
import math
import cplex
import matplotlib.pyplot as plt
from datetime import date
import ipopt
import copy
class erc(object):
    def __init__(self):
        pass

    def objective(self, x):
        # The callback for calculating the objective
        y = x * np.dot(Q, x)
        fval = 0
        for i in range(N):
            for j in range(i,N):
                xij = y[i] - y[j]
                fval = fval + xij*xij
        fval = 2*fval
        return fval

    def gradient(self, x):
        # The callback for calculating the gradient
        grad = np.zeros(N)
        # Insert your gradient computations here
        # You can use finite differences to check the gradient
        y = x * np.dot(Q,x)
        for i in range(N):
            for j in range(i + 1, N):
                if i != j:
                    for grad_i in range(N):
                        if i == grad_i:
                            grad[grad_i] += 2 * (y[i] - y[j]) * (y[i] + x[i] * Q[i,i] - x[j] * Q[j,i])
                        elif j == grad_i:
                            grad[grad_i] += 2 * (y[j] - y[i]) * (y[j] + x[j] * Q[j,j] - x[i] * Q[i,j])
                        else:
                            grad[grad_i] += 2 * (y[i] - y[j]) * (x[i] * Q[i,grad_i] - x[j] * Q[j,grad_i])
        grad = grad * 2
        return grad

    def constraints(self, x):
        # The callback for calculating the constraints
        return [1.0] * N
    
    def jacobian(self, x):
        # The callback for calculating the Jacobian
        return np.array([[1.0] * N])

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        # Example for the use of the intermediate callback. uncomment next line for print
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
        return

# Complete the following functions
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
    x_optimal = x_init
    cash_optimal = cash_init
    return x_optimal, cash_optimal

def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    # compute weights of assets in portfolio
    weights = np.array([1/N] * N)

    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost
    return x_optimal, cash_optimal

def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    # optimization for min variance to derive optimal weights of assets in portfolio
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0] * N
    ub = [cplex.infinity] * N
    lb = [0] * N
    cols = [[[0],[1]]] * N
    cpx.linear_constraints.add(rhs = [1.0], senses = "E")
    var_names = ["w%s" % i for i in range(1,N + 1)] 
    cpx.variables.add(obj = c, lb = lb, ub = ub, columns = cols, names = var_names)
    sparse_indices = [i for i in range(N)]
    qmat = [[sparse_indices, i] for i in (2 * Q).tolist()]
    cpx.objective.set_quadratic(qmat)
    alg = cpx.parameters.lpmethod.values
    cpx.parameters.qpmethod.set(alg.concurrent)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.solve()
    weights = np.array(cpx.solution.get_values())
  
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost

    return x_optimal, cash_optimal, weights

def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    # optimization for max Sharpe ratio to derive optimal weights of assets in portfolio
    negativemu = 0
    for i in range(len(mu)):
        if mu[i] < 0:
            negativemu += 1
    if negativemu == 20:
        return x_init, cash_init
    rf_daily = r_rf2008_2009/252
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0] * (N + 1)
    ub = [cplex.infinity] * (N + 1)
    lb = [0] * (N + 1)
    cols = [[[0,1],[mu[i] - rf_daily,1]] for i in range(N)]
    cols.append([[0,1],[0,-1]])
    cpx.linear_constraints.add(rhs = [1.0,0], senses = "EE")
    var_names = ["y%s" % i for i in range(1,N + 2)] 
    cpx.variables.add(obj = c, lb=lb, ub = ub, columns = cols, names = var_names)
    sparse_indices = [i for i in range(N)]
    qmat = [[sparse_indices, i] for i in (2*Q).tolist()]
    qmat.append([[],[]])
    cpx.objective.set_quadratic(qmat)
    alg = cpx.parameters.lpmethod.values
    cpx.parameters.qpmethod.set(alg.concurrent)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.solve()
    ys = np.array(cpx.solution.get_values())
    weights = ys[:N] / ys[N]

    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost

    return x_optimal, cash_optimal

def strat_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices):
    # optimization for ERC to derive weights of assets in portfolio
    # Use "1/n portfolio" w0 as initial portfolio for starting IPOPT optimization
    lb = [0.0] * N  # lower bounds on variables
    ub = [1.0] * N  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints
    w0 = [1.0/N] * N

    # Define IPOPT problem
    nlp = ipopt.problem(n=len(w0), m=len(cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu)
     
    # Set the IPOPT options
    nlp.addOption('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8'))
    nlp.addOption('hessian_approximation'.encode('utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.addOption('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.addOption('tol'.encode('utf-8'), 1e-10)

    # Solve the problem
    weights, info = nlp.solve(w0)
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost

    return x_optimal, cash_optimal, weights

def strat_lever_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices, erc_weights, rf_asset_value_init):
    # assumption: no transaction cost associated with shorting with risk-free interest
    portf_value_init = x_init.dot(cur_prices) + cash_init - rf_asset_value_init

    # inherit weights from erc portfolio
    weights = erc_weights

    # compute the portfolio before reborrowing
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init - rf_asset_value_init 
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal_1 = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal_1 - x_init).dot(cur_prices) * 0.005
    cash_optimal_1 = portf_value_init - x_optimal_1.dot(cur_prices) - trans_cost

    # money to be borrowed this period
    rf_asset_value_cur = x_optimal_1.dot(cur_prices) + cash_optimal_1

    # compute the portfolio after reborrowing
    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_optimal_1.dot(cur_prices) + cash_optimal_1 + rf_asset_value_cur
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_optimal_1).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal_2 = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal_2 - x_optimal_1).dot(cur_prices) * 0.005
    cash_optimal_2 = portf_value_init - x_optimal_2.dot(cur_prices) - trans_cost
    
    return x_optimal_2, cash_optimal_2, rf_asset_value_cur

def strat_robust_optim(x_init, cash_init, mu, Q, cur_prices, w_minVar):
    # Define initial portfolio ("equally weighted" or "1/n portfolio")
    w0 = [1.0/N] * N
    
    # Sanity check
    Sum_w = sum(w0)
    
    # 1/n portfolio return
    ret_init = np.dot(mu, w0)
    
    # 1/n portfolio variance
    var_init = np.dot(w0, np.dot(Q, w0))

    # Required portfolio robustness
    var_matr = np.diag(np.diag(Q))

    # Target portfolio return estimation error is return estimation error of 1/n portfolio
    rob_init = np.dot(w0, np.dot(var_matr, w0)) # return estimation error of initial portfolio
    rob_bnd  = rob_init # target return estimation error
    
    var_minVar = np.dot(w_minVar, np.dot(Q, w_minVar))
    ret_minVar = np.dot(mu, w_minVar)
    rob_minVar = np.dot(w_minVar, np.dot(var_matr, w_minVar))
    
    # Target portfolio return is 20% more (30% has no solution) than return of minimum variance portfolio
    Portf_Retn = ret_minVar * 1.2
    
    Qq_rMV = var_matr
    Qq_rMVs = np.sqrt(Qq_rMV)
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c  = [0.0] * N
    lb = [0.0] * N
    ub = [1.0] * N
    A = []
    for k in range(N):
        A.append([[0,1],[1.0,mu[k]]])
   
    var_names = ["w_%s" % i for i in range(1,N+1)]
    cpx.linear_constraints.add(rhs=[1.0,Portf_Retn], senses="EG")
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(N)), list(2*Q[k,:])] for k in range(N)]
    cpx.objective.set_quadratic(Qmat)
    Qcon = cplex.SparseTriple(ind1=var_names, ind2=range(N), val=np.diag(var_matr))
    cpx.quadratic_constraints.add(rhs=rob_bnd, quad_expr=Qcon, name="Qc")
    cpx.parameters.threads.set(4)
    cpx.parameters.timelimit.set(60)
    cpx.parameters.barrier.qcpconvergetol.set(1e-12)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_log_stream(None)
    cpx.set_error_stream(None)
    cpx.solve()
    w_rMV = cpx.solution.get_values()
    card_rMV = np.count_nonzero(w_rMV)
    ret_rMV  = np.dot(mu, w_rMV)
    var_rMV = np.dot(w_rMV, np.dot(Q, w_rMV))
    rob_rMV = np.dot(w_rMV, np.dot(var_matr, w_rMV))
    # Round near-zero portfolio weights
    w_rMV = np.array(w_rMV)
    w_rMV_nonrnd = copy.deepcopy(w_rMV)
    w_rMV[w_rMV<1e-6] = 0
    w_rMV = w_rMV / np.sum(w_rMV)

    weights = w_rMV

    # compute upper bound of transaction fee as its estimation and adjust the total porfolio value accordingly
    portf_value_init = x_init.dot(cur_prices) + cash_init
    x_optimal_pre = np.array([int(i) for i in (portf_value_init * weights / cur_prices)])
    trans_cost_est = abs(x_optimal_pre - x_init).dot(cur_prices) * 0.005
    portf_value_adjust = portf_value_init - trans_cost_est

    # compute shares in portfolio and amount in cash account
    x_optimal = np.array([int(i) for i in (portf_value_adjust * weights / cur_prices)])
    trans_cost = abs(x_optimal - x_init).dot(cur_prices) * 0.005
    cash_optimal = portf_value_init - x_optimal.dot(cur_prices) - trans_cost
    return x_optimal, cash_optimal

# Input file
input_file_prices = 'Daily_closing_prices20082009.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)

# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])
# Find the number of trading days in Nov-Dec 2007 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2007)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)

# Remove datapoints for year 2007
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]

# Initial positions in the portfolio
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20000])

# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))

# Initial portfolio weights
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2008-2009 is 2.5%
r_rf = 0.025
# Annual risk-free rate for years 2008-2009 is 4.5%
r_rf2008_2009 = 0.045

# Create a list recording prices at beginning of periods
period_prices = []

# Create a list recording maximum drawdown in periods
max_drawdown = []

# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe', 'strat_equal_risk_contr', 'strat_lever_equal_risk_contr', 'strat_robust_optim']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Mininum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio', 'Equal Risk Contributions Portfolio', 'Leveraged Equal Risk Contributions Portfolio', 'Robust Optimization Portfolio']
# N_strat = 7  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe, strat_equal_risk_contr, strat_lever_equal_risk_contr, strat_robust_optim]

portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
rf_asset_value = 0
for period in range(1, N_periods+1):
    # Compute current year and month, first and last day of the period
    if dates_array[0, 0] == 19:
        cur_year  = 19 + math.floor(period/7)
    else:
        cur_year  = 2008 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
    print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   
    # Prices for the current day
    cur_prices = data_prices[day_ind_start,:]
    period_prices.append(cur_prices)

    # Create a list recording maximum drawdown in each period
    max_drawdown_period = []

    # Execute portfolio selection strategies
    for strategy  in range(N_strat):

        # Get current portfolio positions
        if period == 1:
          curr_positions = init_positions
          curr_cash = 0
          portf_value[strategy] = np.zeros((N_days, 1))
        else:
          curr_positions = x[strategy, period-2]
          curr_cash = cash[strategy, period-2]

        # Compute strategy
        # additional set of weights received from erc portfolio function, then pass the weights to leveraged erc, receiving additional record of risk-free asset amount
        if strategy == 4: 
          x[strategy, period-1], cash[strategy, period-1], erc_weights = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        elif strategy == 5:
          x[strategy, period-1], cash[strategy, period-1], rf_asset_value = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices, erc_weights, rf_asset_value)
        elif strategy == 2:
          x[strategy, period-1], cash[strategy, period-1], w_minVar = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        elif strategy == 6:
          x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices, w_minVar)
        else:
          x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
        # Check that cash account is >= 0
        # Check that we can buy new portfolio subject to transaction costs

        ###################### Insert your code here ############################
        position_pre = curr_positions
        position_post = x[strategy, period - 1]

        # balance of shares sold deducting shares purchased 
        balance = (position_pre - position_post).dot(cur_prices)
    
        # transaction cost
        trans_cost = abs(position_pre - position_post).dot(cur_prices) * 0.005

        # identifying if cash account is nonnegative
        flag_cash_nonneg =  cash[strategy, period - 1] >= 0

        # identifying if the new portfolio and the associated transaction cost can be covered fully by the budget
        if strategy == 5:
          flag_enough_budget = (curr_cash + rf_asset_value) >= trans_cost + ((position_pre + position_post)/2).dot(cur_prices) * 0.005

        else:
          flag_enough_budget = (curr_cash + balance) >= trans_cost

        # verification of portfolio feasibility and print a message if the portfolio is infeasible
        if not(flag_cash_nonneg) or not(flag_enough_budget):
          print()
          print("Strategy is infeasible.")
          print()

        # Compute portfolio value
        p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
        portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))

        #subtract money borrowed and its interest generated from portfolio value
        if strategy ==5:
          for i in range(day_ind_start, day_ind_end+1):
            portf_value[5][i] -= rf_asset_value * (1 + r_rf2008_2009 / 6 * (i - day_ind_start) / (day_ind_end - day_ind_start))
        
        print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
              portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))
        
        # include risk-free interest for 2 months (one period)
        rf_asset_value *= (1 + r_rf2008_2009 / 6)


        max_drawdown_period.append((max(portf_value[strategy][day_ind_start:day_ind_end]) - min(portf_value[strategy][day_ind_start:day_ind_end]))/max(portf_value[strategy][day_ind_start:day_ind_end])[0])

    # Compute expected returns and covariances for the next period
    cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
    mu = np.mean(cur_returns, axis = 0)
    Q = np.cov(cur_returns.T)
    max_drawdown.append(max_drawdown_period)
# Plot results
###################### Insert your code here ############################

# Plot of portfolio value for different strategies
df_daily_portf_value = pd.DataFrame(np.array([i.flatten() for i in portf_value]).T, index = pd.DatetimeIndex(dates), columns = strategy_names)
df_daily_portf_value.plot()
start_date = date.fromisoformat('2008-01-02')
end_date = date.fromisoformat('2009-12-31')
plt.xlim([start_date, end_date])
plt.title('Portfolio Value for Different Strategies from 2008 to 2009', fontweight="bold", pad = 20, fontsize="12")
plt.xlabel('Date', fontweight="bold", fontsize="12")
plt.ylabel('Portfolio Value', fontweight="bold", fontsize="12")
plt.show()

# Plot of maximum drawdown for different strategies
df_max_drawdown = pd.DataFrame(np.array([i.flatten() for i in np.array(max_drawdown)]), index = [i for i in range(1,13)], columns = strategy_names)
df_max_drawdown.plot()
plt.title('Maximum Drawdown (absolute value) for Different Strategies from 2008 to 2009', fontweight="bold", pad = 20, fontsize="12")
plt.xlabel('Period', fontweight="bold", fontsize="12")
plt.ylabel('Maximum Drawdown', fontweight="bold", fontsize="12")
plt.show()

trategy_ids = [3,4,7]

# Plot of dynamic changes (in terms of weights of value) in portfolio allocation for strategy 3, 4, 7
# get name list of stocks
stock_list = df.columns.tolist()[1:]
weights_data = []
for strategy  in [2,3,6]:
    stock_data = {}
    for stock in range(N):
        weights_period = []
        for period in range(N_periods):
                weights_period.append(x[strategy, period][stock] * period_prices[period][stock] / (x[strategy, period].dot(period_prices[period]) + cash[strategy, period]))
        stock_data[stock_list[stock]] = weights_period
    weights_data.append(stock_data)

periods = [i for i in range(1, N_periods + 1)]

color_map = ["#9ACD32", "#00FF7F", "#EE82EE", "#40E0D0", "#FF6347", "#4682B4", "#6A5ACD","#000080", "#FF0000", "#FFFFE0", "#000000", "#696969", "#800080","#FFFF00","#B8860B","#E0FFFF","#008000","#FF69B4", "#0000CD", "#A52A2A"]
for i in range(3):
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 12])
    ax.stackplot(periods, weights_data[i].values(),labels=weights_data[i].keys(),colors = color_map)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.set_title('Portfolio Composition vs. Periods for Strategy %s'%(trategy_ids[i]), fontweight="bold", fontsize="12")
    ax.set_xlabel('Period', fontweight="bold", fontsize="12")
    ax.set_ylabel('Value Weights (Accumulated)', fontweight="bold", fontsize="12")
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.show()

# Plot of dynamic changes (in terms of weights of position) in portfolio allocation for strategy 7
# get name list of stocks
stock_list = df.columns.tolist()[1:]
weights_data = []
for strategy  in [2,3,6]:
    stock_data = {}
    for stock in range(N):
        weights_period = []
        for period in range(N_periods):
                weights_period.append(x[strategy, period][stock] / sum(x[strategy, period]))
        stock_data[stock_list[stock]] = weights_period
    weights_data.append(stock_data)

periods = [i for i in range(1, N_periods + 1)]

color_map = ["#9ACD32", "#00FF7F", "#EE82EE", "#40E0D0", "#FF6347", "#4682B4", "#6A5ACD","#000080", "#FF0000", "#FFFFE0", "#000000", "#696969", "#800080","#FFFF00","#B8860B","#E0FFFF","#008000","#FF69B4", "#0000CD", "#A52A2A"]
for i in range(3):
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    ax.set_xlim([1, 12])
    ax.stackplot(periods, weights_data[i].values(),labels=weights_data[i].keys(),colors = color_map)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.set_title('Portfolio Composition vs. Periods for Strategy %s'%(trategy_ids[i]), fontweight="bold", fontsize="12")
    ax.set_xlabel('Period', fontweight="bold", fontsize="12")
    ax.set_ylabel('Position Weights (Accumulated)', fontweight="bold", fontsize="12")
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.show()