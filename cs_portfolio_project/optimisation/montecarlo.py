import numpy as np
import pandas as pd

from typing import Callable, Dict, Any, Optional
import matplotlib.pyplot as plt


def find_portfolio_weights_and_value(skins: list, quantity: list, asset_prices: pd.DataFrame):
    """
    Calculate portfolio weights and total portfolio value based on asset holdings (skin inventory).

    args:
        skins (List): asset names (column names in asset_prices) representing the portfolio.
        quantity (List): listCorresponding list of quantities held for each asset in 'skins'.  
        asset_prices (df): historical price data for all assets.

    Returns:   
        weights: portfolio weights for all assets in asset_prices. Assets not in 'skins' will have zero weight.
        portofolio_value : Total market value of the portfolio, based on the latest available (non-NaN) prices.
    """

    if len(skins) != len(quantity):
        raise ValueError("Length of 'skins' and 'quantity' are not equal")

    latest_prices = asset_prices[skins].ffill().iloc[-1]
    capital = latest_prices * quantity
    portofolio_value = capital.sum()
    weights_caps = capital/capital.sum()
    weights = pd.Series(0.0, index=asset_prices.columns)
    weights[weights_caps.index] = weights_caps

    return weights, portofolio_value


def find_portfolio_weights(skins: list, quantity: list, asset_prices: pd.DataFrame):
    return find_portfolio_weights_and_value(skins, quantity, asset_prices)[0]


def find_portfolio_value(skins: list, quantity: list, asset_prices: pd.DataFrame):
    return find_portfolio_weights_and_value(skins, quantity, asset_prices)[1]


def monte_carlo_simulation(expected_rets, cov_matrix, weights, initial_portfolio_value, number_of_sims=100, sim_timeframe=365, log=True):

    rets = expected_rets.values
    cov = cov_matrix.values

    # n_assets = expected_rets.shape[0]
    n_assets = len(rets)
    portfolio_sims = np.zeros((sim_timeframe, number_of_sims))
    L = np.linalg.cholesky(cov)

    for m in range(number_of_sims):
        Z = np.random.normal(size=(sim_timeframe, n_assets))
        correlated_returns = Z @ L.T + rets  # shape: (T, n_assets)
        portfolio_returns = correlated_returns @ weights
        if log:
            portfolio_prices = initial_portfolio_value * \
                np.exp(np.cumsum(portfolio_returns))
        else:
            portfolio_prices = initial_portfolio_value * \
                np.cumprod(1 + portfolio_returns)

        portfolio_sims[:, m] = portfolio_prices
    return portfolio_sims


def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """

    return np.percentile(returns, alpha)


def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    belowVaR = returns <= mcVaR(returns, alpha=alpha)
    return returns[belowVaR].mean()


def max_drawdown(portfolio_path):
    """
    Calculate maximum drawdown from a single portfolio simulation path.

    Parameters:
        portfolio_path (ndarray): 1D array of portfolio values

    Returns:
        float: max drawdown as a positive decimal (e.g., 0.25 means 25%)
    """
    running_max = np.maximum.accumulate(portfolio_path)
    drawdowns = 1 - portfolio_path / running_max
    return np.max(drawdowns)


def plot_simulation_results(portfolio_sims):
    final_values = portfolio_sims[-1]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(portfolio_sims)
    axs[0].set_title('Monte Carlo Simulation Paths')
    axs[0].set_xlabel('Days')
    axs[0].set_ylabel('Portfolio Value ($)')
    axs[1].hist(final_values, bins=50, color='skyblue', edgecolor='black')
    axs[1].set_title('Distribution of Final Portfolio Values')
    axs[1].set_xlabel('Final Portfolio Value ($)')
    axs[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def simulate_portfolio_performance(
    rets: pd.DataFrame,
    weight_func: Callable,
    initial_portfolio_value,
    number_of_sims=100,
    sim_timeframe=365,
    log=True,
    expected_returns_func: Optional[Callable] = None,
    covariance_func: Optional[Callable] = None,
    expected_returns_kwargs: dict = {},
    covariance_kwargs: dict = {},
    weight_func_kwargs: dict = {}
):
    """
    Monte carlo simulation for custom weight, covariance and expected_returns functions.

    Args:
        rets (pd.DataFrame): Historical asset returns.
        weight_func (Callable): Function that returns portfolio weights.
        initial_portfolio_value (float): Starting value of the portfolio.
        number_of_sims (int): Number of simulation paths. Default is 100.
        sim_timeframe (int): Number of days to simulate. Default is 365.
        log (bool): If True, uses log-normal returns. Otherwise, uses simple returns.
        expected_returns_func (Callable, optional): Custom function to compute expected returns.
        covariance_func (Callable, optional): Custom function to compute the covariance matrix.
        expected_returns_kwargs (dict): Extra arguments for `expected_returns_func`.
        covariance_kwargs (dict): Extra arguments for `covariance_func`.
        weight_func_kwargs (dict): Extra arguments for `weight_func`.

    Returns:
        tuple:
            - np.ndarray: Simulated portfolio paths.
            - dict: Portfolio statistics with:
                - 'mean_portfolio_value'
                - 'median_portfolio_value'
                - 'standard_deviation'
                - 'value_at_risk_5_pct'
                - 'conditional_VAR_5_pct'
                - 'probability_of_loss'

    Notes:
        This function also visualizes:
        - All simulated portfolio paths.
        - Histogram of final simulated portfolio values.
    """
    rets_copy = rets.copy()
    expected_returns = None
    if expected_returns_func:
        expected_returns = expected_returns_func(
            rets_copy, **expected_returns_kwargs)
    else:
        expected_returns = rets_copy.mean()

    cov_matrix = None
    if covariance_func:
        cov_matrix = covariance_func(rets_copy, **covariance_kwargs)
    else:
        cov_matrix = rets_copy.cov()

    weights = weight_func(**weight_func_kwargs)

    portfolio_sims = monte_carlo_simulation(
        expected_returns, cov_matrix, weights, initial_portfolio_value, number_of_sims, sim_timeframe, log)
    portfolio_sims_last = portfolio_sims[-1, :]
    mean_portfolio_value = portfolio_sims_last.mean()
    median_portfolio_value = np.median(portfolio_sims_last)
    std_portfolio = portfolio_sims_last.std()
    conditional_VAR_5_pct = mcCVaR(portfolio_sims_last, 5)
    value_at_risk_5_pct = np.percentile(portfolio_sims_last, 5)
    prob_loss = np.mean(portfolio_sims_last < initial_portfolio_value)

    sim_information = {
        'mean_portfolio_value': mean_portfolio_value,
        'median_portfolio_value': median_portfolio_value,
        'standard_deviation': std_portfolio,
        'value_at_risk_5_pct': value_at_risk_5_pct,
        'conditional_VAR_5_pct': conditional_VAR_5_pct,
        "probability_of_loss": prob_loss
    }
    plot_simulation_results(portfolio_sims)
    return portfolio_sims, sim_information

# ['fracture','shadow','huntsman_weapon','falchion'],[200,100,10,40]
