# Import libraries
import pandas as pd
import numpy as np
import math
import cplex
import matplotlib.pyplot as plt
from datetime import date

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
	cpx.set_log_stream(None)
	cpx.set_results_stream(None)
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

	return x_optimal, cash_optimal

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
	cpx.set_log_stream(None)
	cpx.set_results_stream(None)
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

# Create a list recording prices at beginning of periods
period_prices = []

# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Mininum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio']
#N_strat = 1  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe]

portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
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
		flag_enough_budget = (curr_cash + balance) >= trans_cost

		# verification of portfolio feasibility and print a message if the portfolio is infeasible
		if not(flag_cash_nonneg) or not(flag_enough_budget):
			print()
			print("Strategy is infeasible.")
			print()

		# Compute portfolio value
		p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
		portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
		print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
			portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))

      
	# Compute expected returns and covariances for the next period
	cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
	mu = np.mean(cur_returns, axis = 0)
	Q = np.cov(cur_returns.T)

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

# Plot of dynamic changes in portfolio allocation for strategy 3 and 4
# get name list of stocks
stock_list = df.columns.tolist()[1:]
weights_data = []
for strategy  in range(2, N_strat):
	stock_data = {}
	for stock in range(N):
		weights_period = []
		for period in range(N_periods):
				weights_period.append(x[strategy, period][stock] * period_prices[period][stock] / (x[strategy, period].dot(period_prices[period]) + cash[strategy, period]))
		stock_data[stock_list[stock]] = weights_period
	weights_data.append(stock_data)

periods = [i for i in range(1, N_periods + 1)]

color_map = ["#9ACD32", "#00FF7F", "#EE82EE", "#40E0D0", "#FF6347", "#4682B4", "#6A5ACD","#000080", "#FF0000", "#FFFFE0", "#000000", "#696969", "#800080","#FFFF00","#B8860B","#E0FFFF","#008000","#FF69B4", "#0000CD", "#A52A2A"]
for i in range(2):
	fig, ax = plt.subplots()
	ax.set_ylim([0, 1])
	ax.set_xlim([1, 12])
	ax.stackplot(periods, weights_data[i].values(),labels=weights_data[i].keys(),colors = color_map)
	ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
	ax.set_title('Portfolio Composition vs. Periods for Strategy %s'%(i + 3), fontweight="bold", fontsize="12")
	ax.set_xlabel('Period', fontweight="bold", fontsize="12")
	ax.set_ylabel('Weights (Accumulated)', fontweight="bold", fontsize="12")
	plt.rcParams["figure.figsize"] = [10, 6]
	plt.show()

# Part 3 Variation
# Repeat the computation of portfolio value process for variation of strategies with holding portfolio 2 periods and 2 years
##################################################
portf_value_variations = []
for variation in range(2):
	mu = np.mean(cur_returns0, axis = 0)
	Q = np.cov(cur_returns0.T)
	portf_value_variation = [0] * N_strat
	x = np.zeros((N_strat, N_periods), dtype=np.ndarray)
	cash = np.zeros((N_strat, N_periods), dtype=np.ndarray)
	for period in range(1, N_periods + 1):
		# Compute current year and month, first and last day of the period
		if dates_array[0, 0] == 19:
			cur_year = 19 + math.floor(period / 7)
		else:
			cur_year = 2019 + math.floor(period / 7)

		cur_month = 2 * ((period - 1) % 6) + 1
		day_ind_start = min(
			[i for i, val in enumerate((dates_array[:, 0] == cur_year) & (dates_array[:, 1] == cur_month)) if val])
		day_ind_end = max(
			[i for i, val in enumerate((dates_array[:, 0] == cur_year) & (dates_array[:, 1] == cur_month + 1)) if val])

		# Prices for the current day
		cur_prices = data_prices[day_ind_start, :]
		period_prices.append(cur_prices)

		# Execute portfolio selection strategies
		for strategy in range(1, N_strat):

			# Get current portfolio positions
			if period == 1:
				curr_positions = init_positions
				curr_cash = 0
				portf_value_variation[strategy] = np.zeros((N_days, 1))
			else:
				curr_positions = x[strategy, period - 2]
				curr_cash = cash[strategy, period - 2]

			# Compute strategy
			if variation == 0:
				if period == 1:
					x[strategy, period - 1], cash[strategy, period - 1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
				else:
					x[strategy, period - 1], cash[strategy, period - 1] = fh_array[0](curr_positions, curr_cash, mu, Q, cur_prices)
			else:
				if period % 2 == 1:
					x[strategy, period - 1], cash[strategy, period - 1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
				else:
					x[strategy, period - 1], cash[strategy, period - 1] = fh_array[0](curr_positions, curr_cash, mu, Q, cur_prices)

			# Verify that strategy is feasible (you have enough budget to re-balance portfolio)
			# Check that cash account is >= 0
			# Check that we can buy new portfolio subject to transaction costs
			position_pre = curr_positions
			position_post = x[strategy, period - 1]

			# balance of shares sold deducting shares purchased
			balance = (position_pre - position_post).dot(cur_prices)

			# transaction cost
			trans_cost = abs(position_pre - position_post).dot(cur_prices) * 0.005

			# identifying if cash account is nonnegative
			flag_cash_nonneg = cash[strategy, period - 1] >= 0

			# identifying if the new portfolio and the associated transaction cost can be covered fully by the budget
			flag_enough_budget = (curr_cash + balance) >= trans_cost

			# verification of portfolio feasibility and print a message if the portfolio is infeasible
			if not (flag_cash_nonneg) or not (flag_enough_budget):
				print()
				print("Strategy is infeasible.")
				print()

			# Compute portfolio value
			p_values = np.dot(data_prices[day_ind_start:day_ind_end + 1, :], x[strategy, period - 1]) + cash[
				strategy, period - 1]
			portf_value_variation[strategy][day_ind_start:day_ind_end + 1] = np.reshape(p_values, (p_values.size, 1))

		# Compute expected returns and covariances for the next period
		cur_returns = data_prices[day_ind_start + 1:day_ind_end + 1, :] / data_prices[day_ind_start:day_ind_end, :] - 1
		mu = np.mean(cur_returns, axis=0)
		Q = np.cov(cur_returns.T)
	portf_value_variations.append(portf_value_variation)

# Plot charts comparing the portfolio of each strategy and its corresponding variation strategy
for strategy in range(1, 4):
	df_daily_portf_value = pd.DataFrame(np.array([portf_value[strategy].flatten(),portf_value_variations[0][strategy].flatten(),portf_value_variations[1][strategy].flatten()]).T, index = pd.DatetimeIndex(dates), columns = [strategy_names[strategy],strategy_names[strategy]+'_hold_2_years', strategy_names[strategy]+'_hold_2_periods'])
	df_daily_portf_value.plot()
	start_date = date.fromisoformat('2019-01-02')
	end_date = date.fromisoformat('2020-12-31')
	plt.xlim([start_date, end_date])
	plt.title('Portfolio Value Comparison for Strategy and it Variation from 2019 to 2020', fontweight="bold", pad = 20, fontsize="12")
	plt.xlabel('Date', fontweight="bold", fontsize="12")
	plt.ylabel('Portfolio Value', fontweight="bold", fontsize="12")
	plt.show()
