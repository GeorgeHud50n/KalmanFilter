# Kalman Filter for Statistical Arbitrage

This repository contains Python code that demonstrates the use of the Kalman Filter for statistical arbitrage. The code applies the Kalman Filter to a pair of currency exchange rates, EUR/USD and GBP/USD, to identify trading opportunities based on the spread between the two rates.

The main features of the code include:

- Downloading historical data for the currency exchange rates from Yahoo Finance using the `yfinance` library.
- Applying the Kalman Filter to estimate the hedge ratio between the two rates.
- Calculating the spread and Z-score of the spread to identify trading signals.
- Applying a trading strategy based on specified entry and exit thresholds.
- Evaluating the performance of the strategy and visualizing the results.

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
