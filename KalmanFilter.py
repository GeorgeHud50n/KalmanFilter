import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

def download_data(tickers, start_date, end_date):
    data = []
    for ticker in tickers:
        try:
            data.append(yf.download(ticker, start=start_date, end=end_date)['Close'])
        except Exception as e:
            print(f"Error occurred while downloading data for {ticker}: {e}")
            return None
    data = pd.concat(data, axis=1)
    data.columns = ['EUR/USD', 'GBP/USD']  # setting column names explicitly
    return data

def apply_kalman_filter(data, trans_cov=0.01):
    kf = KalmanFilter(transition_matrices = [1],
                      observation_matrices = [[1]],
                      initial_state_mean = 0,
                      initial_state_covariance = 1,
                      observation_covariance=1,
                      transition_covariance=trans_cov)
    means = np.zeros(len(data))
    covs = np.zeros(len(data))
    mean = kf.initial_state_mean
    cov = kf.initial_state_covariance
    for t in range(len(data)):
        obs_matrix = np.array([data['GBP/USD'].iat[t]])
        mean, cov = kf.filter_update(filtered_state_mean=mean,
                                     filtered_state_covariance=cov,
                                     observation=data['EUR/USD'].iat[t],
                                     observation_matrix=obs_matrix)
        means[t] = mean
        covs[t] = cov
    data['Hedge Ratio'] = -means
    return data

def calculate_spread_and_z_score(data, window=60):
    data['Spread'] = data['EUR/USD'] - data['Hedge Ratio']*data['GBP/USD']
    data['Mean Spread'] = data['Spread'].rolling(window=window).mean()
    data['Std Spread'] = data['Spread'].rolling(window=window).std()
    data['Z-Score'] = (data['Spread'] - data['Mean Spread']) / data['Std Spread']
    return data

def check_stationarity(data):
    residuals = data['Spread'] - data['Mean Spread']
    p_value = adfuller(residuals.dropna())[1]
    print(f'ADF p-value: {p_value:.5f}')

def apply_trading_strategy(data, entry_threshold, exit_threshold):
    data['Long Entry'] = (data['Z-Score'] < -entry_threshold)
    data['Long Exit'] = (data['Z-Score'] >= exit_threshold)
    data['Short Entry'] = (data['Z-Score'] > entry_threshold)
    data['Short Exit'] = (data['Z-Score'] <= -exit_threshold)
    data['Long Positions'] = np.nan
    data.loc[data['Long Entry'],'Long Positions'] = 1
    data.loc[data['Long Exit'],'Long Positions'] = 0
    data['Short Positions'] = np.nan
    data.loc[data['Short Entry'],'Short Positions'] = -1
    data.loc[data['Short Exit'],'Short Positions'] = 0
    data['Long Positions'].fillna(method='ffill', inplace=True)
    data['Short Positions'].fillna(method='ffill', inplace=True)
    data['Positions'] = data['Long Positions'] + data['Short Positions']
    return data

def plot_strategy(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,7))
    data['Z-Score'].plot(ax=ax1, color='darkblue')
    ax1.fill_between(data.index, -2, 2, color='gray', alpha=0.3)
    ax1.set_title('Z-Score of EUR/USD and GBP/USD Spread Over Time')
    ax1.set_ylabel('Z-Score')
    ax1.grid(True)
    data['Positions'].plot(ax=ax2, color='blue', linestyle='--')
    ax2.set_title('Trading Positions Over Time')
    ax2.set_ylabel('Position')
    ax2.set_xlabel('Date')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_returns(data):
    data['EUR/USD Returns'] = data['EUR/USD'].pct_change()
    data['GBP/USD Returns'] = data['GBP/USD'].pct_change()
    data['Strategy Returns'] = data['EUR/USD Returns'] * data['Positions'] - data['GBP/USD Returns'] * data['Positions']
    data['Cumulative Strategy Returns'] = (1 + data['Strategy Returns']).cumprod()
    return data

def plot_returns(data):
    plt.figure(figsize=(12,6))
    data['Cumulative Strategy Returns'].plot()
    plt.title('Cumulative Strategy Returns')
    plt.show()

def sensitivity_analysis(data, transition_covs, window_sizes, entry_thresholds, exit_thresholds):
    results = []
    for trans_cov in transition_covs:
        for window in window_sizes:
            for entry_threshold in entry_thresholds:
                for exit_threshold in exit_thresholds:
                    data_copy = data.copy()
                    data_copy = apply_kalman_filter(data_copy, trans_cov)
                    data_copy = calculate_spread_and_z_score(data_copy, window)
                    data_copy = apply_trading_strategy(data_copy, entry_threshold, exit_threshold)
                    data_copy = calculate_returns(data_copy)
                    total_return = data_copy['Cumulative Strategy Returns'].iloc[-1]
                    results.append((trans_cov, window, entry_threshold, exit_threshold, total_return))
    results_df = pd.DataFrame(results, columns=['Transition Covariance', 'Window Size', 'Entry Threshold', 'Exit Threshold', 'Total Return'])
    return results_df

def main():
    start_date = '2018-06-11'
    end_date = '2023-06-11'
    data = download_data(['EURUSD=X', 'GBPUSD=X'], start_date, end_date)

    # Run sensitivity analysis
    transition_covs = np.linspace(0.01, 1, 5)
    window_sizes = np.linspace(30, 90, 5, dtype=int)
    entry_thresholds = np.linspace(0.05, 0.15, 5)
    exit_thresholds = np.linspace(0.05, 0.15, 5)
    results_df = sensitivity_analysis(data, transition_covs, window_sizes, entry_thresholds, exit_thresholds)

    # Find the best parameters
    best_params = results_df.loc[results_df['Total Return'].idxmax()]
    best_trans_cov = best_params['Transition Covariance']
    best_window_size = int(best_params['Window Size'])  # make sure it's an integer
    best_entry_threshold = best_params['Entry Threshold']
    best_exit_threshold = best_params['Exit Threshold']

    # 3D plotting
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(14, 7))

    # Transition Covariance and Window Size
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('Transition Covariance')
    ax1.set_zlabel('Total Return')
    pivot_table = results_df[results_df['Entry Threshold'] == 0.05][results_df['Exit Threshold'] == 0.05].pivot('Transition Covariance', 'Window Size', 'Total Return')
    X1, Y1 = np.meshgrid(pivot_table.columns, pivot_table.index)
    surf1 = ax1.plot_surface(X1, Y1, pivot_table.values, cmap='viridis')

    # Entry Threshold and Exit Threshold
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('Entry Threshold')
    ax2.set_ylabel('Exit Threshold')
    ax2.set_zlabel('Total Return')
    pivot_table = results_df[results_df['Transition Covariance'] == 0.01][results_df['Window Size'] == 30].pivot('Entry Threshold', 'Exit Threshold', 'Total Return')
    X2, Y2 = np.meshgrid(pivot_table.columns, pivot_table.index)
    surf2 = ax2.plot_surface(X2, Y2, pivot_table.values, cmap='viridis')

    plt.show()

    # Run the strategy using the best parameters
    data = apply_kalman_filter(data, best_trans_cov)
    data = calculate_spread_and_z_score(data, best_window_size)
    check_stationarity(data)
    data = apply_trading_strategy(data, best_entry_threshold, best_exit_threshold)
    plot_strategy(data)
    data = calculate_returns(data)
    plot_returns(data)

    print(f"Best Transition Covariance: {best_trans_cov}")
    print(f"Best Window Size: {best_window_size}")
    print(f"Best Entry Threshold: {best_entry_threshold}")
    print(f"Best Exit Threshold: {best_exit_threshold}")
    print(f"Total Return with Best Parameters: {best_params['Total Return']}")

if __name__ == '__main__':
    main()
