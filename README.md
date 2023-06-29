# Kalman Filter for Statistical Arbitrage

This repository contains Python code that demonstrates the use of the Kalman Filter for statistical arbitrage. The code applies the Kalman Filter to a pair of currency exchange rates, EUR/USD and GBP/USD, to identify trading opportunities based on the spread between the two rates.

The main features of the code include:

- Downloading historical data for the currency exchange rates from Yahoo Finance using the `yfinance` library.
- Applying the Kalman Filter to estimate the hedge ratio between the two rates.
- Calculating the spread and Z-score of the spread to identify trading signals.
- Applying a trading strategy based on specified entry and exit thresholds.
- Evaluating the performance of the strategy and visualizing the results.

# Sensitivity Analysis

The sensitivity analysis in this code performs an exhaustive search over a range of parameters to find the best combination for the trading strategy. However, it's important to note that this approach can be computationally expensive, especially if the parameter ranges are large or the dataset is large.

The complexity of the sensitivity analysis can be attributed to the nested loops that iterate over the parameter ranges. Each iteration involves applying the Kalman Filter, calculating the spread and Z-score, applying the trading strategy, and calculating the returns. As a result, the execution time increases as the number of parameter combinations and the size of the dataset grow.

To mitigate the computational expense, it's advisable to carefully select the parameter ranges based on domain knowledge or prior analysis. Consider narrowing down the parameter ranges to a reasonable and meaningful subset that captures the desired characteristics of the trading strategy.

Additionally, optimizing the code's efficiency can help reduce the computational time. Consider implementing techniques such as vectorization, parallelization, or using more efficient libraries or algorithms where applicable.

It's essential to strike a balance between the granularity of the parameter search and the available computational resources. Consider running the sensitivity analysis on a subset of the parameter combinations initially and gradually expanding the search if necessary.

Remember that the computational expense can vary depending on the specific hardware, software, and dataset used. It's recommended to monitor the execution time and resource usage during the sensitivity analysis to ensure it is within acceptable limits.

## Disclaimer

The sensitivity analysis provided in this code is for demonstration purposes only. The parameter ranges used in this example are arbitrary and may not reflect the optimal values for real-world trading scenarios. It's important to perform further analysis, backtesting, and validation before deploying any trading strategies in live trading environments.


## Prerequisites

Make sure you have the following dependencies installed:

- pandas
- numpy
- pykalman
- statsmodels
- yfinance
- matplotlib
- seaborn

You can install these dependencies using `pip`:

```bash
pip install pandas numpy pykalman statsmodels yfinance matplotlib seaborn
