
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def plot_time_series_decompose(data,model,period=365):
    """
    Plot the time series decomposition ('Observed', 'Trend', 'Seasonal', 'Residual').

    Args:
        data (pd.Series): Time series (can be returns or price).
        model (str): 'additive' or 'multiplicative'.
        period (int): periodicity (1 for annual, 12 for monthly, ect.).

    Note:
        checks for stationarity, if p-value <0.05, time series is likely stationary
    """
    result = seasonal_decompose(data, model=model, period=period)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    components = ['Observed', 'Trend', 'Seasonal', 'Residual']
    plot_data = [result.observed, result.trend, result.seasonal, result.resid]

    for ax, comp, series in zip(axs, components, plot_data):
        ax.plot(series.index, series.values, marker='o', markersize=1, linewidth=1)
        ax.set_title(comp, fontsize=14)
        ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    result = adfuller(data)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])


def plot_ACF_and_PACF(data,lags=20):

    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    plot_acf(data.dropna(), lags=lags, ax=axs[0])
    axs[0].set_title('Autocorrelation (ACF)', fontsize=14)
    axs[0].grid(True)

    plot_pacf(data.dropna(), lags=lags, ax=axs[1])
    axs[1].set_title('Partial Autocorrelation (PACF)', fontsize=14)
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()