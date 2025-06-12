import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from typing import Callable, Dict, Any, Optional
import inspect


def get_alpha_and_beta(rets, market_rets):
    """
    return: dataframe with alpha and beta for all assets
    """

    results = {'Asset': [], 'Alpha': [], 'Beta': []}
    for asset in rets.columns:  # Exclude 'market'
        y = rets[asset]
        X = market_rets.rename("market")

        # Drop rows where either y or X is NaN
        valid_data = pd.concat([y, X], axis=1).dropna()
        if len(valid_data) < 2:
            results['Asset'].append(asset)
            results['Alpha'].append(np.nan)
            results['Beta'].append(np.nan)
            continue

        X_valid = sm.add_constant(valid_data['market'])
        y_valid = valid_data[asset]

        model = sm.OLS(y_valid, X_valid).fit()

        results['Asset'].append(asset)
        results['Alpha'].append(model.params['const'])  # Intercept (alpha)
        results['Beta'].append(model.params['market'])  # Slope (beta)

    results_CAPM = pd.DataFrame(results)
    return results_CAPM


def get_expected_returns_CAPM(rets, market_rets, risk_free_rate=0.0):
    expected_market_return = market_rets.mean()
    capm_results = get_alpha_and_beta(rets, market_rets).set_index('Asset')

    expected_returns = pd.Series({
        asset: risk_free_rate + capm_results.loc[asset, 'Beta'] * (
            expected_market_return - risk_free_rate) + capm_results.loc[asset, 'Alpha']
        for asset in capm_results.index
    })
    return expected_returns


def plot_alpha_beta(results_CAPM):
    """ plots alpha and beta values for all assets.
    """
    plt.figure(figsize=(10, 6))
    for x in results_CAPM.index:
        asset_alpha = results_CAPM.loc[x, 'Alpha']
        asset_beta = results_CAPM.loc[x, 'Beta']
        plt.scatter([asset_alpha], [asset_beta],
                    label=results_CAPM.loc[x, 'Asset'])
        plt.text(asset_alpha, asset_beta,
                 results_CAPM.loc[x, 'Asset'], fontsize=6, ha='left', va='bottom')
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.title('alpha beta scatter plot')
    plt.grid(True)
    plt.show()


def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def minimize_volatility(weights, cov_matrix):
    return portfolio_volatility(weights, cov_matrix)


def maximize_rets_for_eq_vol(weights, cov_matrix):
    return portfolio_return(weights, cov_matrix)


def get_efficient_frontier(rets, market_rets, cov_matrix, n_points=50, risk_free_rate=0.0):
    # cov_matrix = returns_and_mkt.drop(columns='market').cov()
    expected_returns = get_expected_returns_CAPM(
        rets, market_rets, risk_free_rate)

    n_assets = len(expected_returns)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum weights = 1
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # No short
    initial_weights = np.array([1/n_assets] * n_assets)

    # Range of target returns
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    efficient_frontier = []
    for target_ret in target_returns:
        constraints.append({'type': 'eq', 'fun': lambda w: portfolio_return(
            w, expected_returns) - target_ret})
        result = minimize(
            minimize_volatility,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            volatility = portfolio_volatility(result.x, cov_matrix)
            efficient_frontier.append((target_ret, volatility))
        constraints.pop()  # Remove the return constraint for the next iteration

    #  arrays for plotting
    eff_returns, eff_volatilities = zip(*efficient_frontier)
    return eff_returns, eff_volatilities


def get_minimum_var_portfolio(rets, market_rets, cov_matrix, risk_free_rate=0.0, days_in_sample=365):
    """ find the minimum variance portofilio compisition
    args:
        days_in_sample: number of days to annualiz
    returns:
        portfolio weights,returns,volatility
    """
    expected_returns = get_expected_returns_CAPM(
        rets, market_rets, risk_free_rate)
    n_assets = len(expected_returns)
    initial_weights = np.array([1/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    gmv_result = minimize(
        minimize_volatility,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    # Extract GMV portfolio details
    gmv_weights = gmv_result.x
    gmv_return = portfolio_return(gmv_weights, expected_returns)
    gmv_volatility = portfolio_volatility(gmv_weights, cov_matrix)
    print(f'weights : {gmv_weights*100}')
    print(f'portfolio return (ann) : {gmv_return*days_in_sample}')
    print(f'portfolio vol (ann): {gmv_volatility*np.sqrt(days_in_sample)}')
    return gmv_weights, gmv_return, gmv_volatility


def plot_efficient_frontier(eff_returns, eff_volatilities, market_vol, market_ret, cov_matrix, expected_returns):
    """
    market_vol = returns['market'].std()
    """

    plt.figure(figsize=(10, 6))
    plt.plot(eff_volatilities, eff_returns, 'b-', label='Efficient Frontier')
    plt.scatter([market_vol], [market_ret], color='red',
                marker='o', label='Market Portfolio')
    for asset in expected_returns.index:
        asset_vol = np.sqrt(cov_matrix.loc[asset, asset])
        asset_ret = expected_returns[asset]
        plt.scatter([asset_vol], [asset_ret], label=asset)
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier with Assets and Market')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_efficient_frontier_2(rets, market_rets, n_points=50, risk_free_rate=0.0):
    """
    market_vol = returns['market'].std()
    """

    eff_returns, eff_volatilities = get_efficient_frontier(
        rets, market_rets, n_points, risk_free_rate)
    market_vol, market_ret = market_rets.std(
    ), market_rets.mean()

    cov_matrix = rets.cov()
    expected_returns = get_expected_returns_CAPM(
        rets, market_rets, risk_free_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(eff_volatilities, eff_returns, 'b-', label='Efficient Frontier')
    plt.scatter([market_vol], [market_ret], color='red',
                marker='o', label='Market Portfolio')
    plt.text(market_vol, market_ret, 'Market',
             fontsize=7, ha='right', va='bottom')
    for asset in expected_returns.index:
        asset_vol = np.sqrt(cov_matrix.loc[asset, asset])
        asset_ret = expected_returns[asset]
        plt.scatter([asset_vol], [asset_ret], label=asset)
        plt.text(asset_vol, asset_ret, asset,
                 fontsize=7, ha='left', va='bottom')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier with Assets and Market')
    plt.grid(True)
    plt.show()


def plot_hierarchical_clustering(returns, correlation_matrix):
    distance_matrix = 1 - correlation_matrix
    linkage_matrix = linkage(distance_matrix, method='ward')
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=returns.columns, leaf_rotation=90)
    plt.title("Hierarchical Clustering of Assets")
    plt.show()


def kmean_clustering(n_clusters, distance_matrix):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(distance_matrix)
    return clusters

def get_equal_weight_pf(rets: pd.DataFrame)->pd.Series:
    """ get the equal weights portfolio variance portofilio composition
    Args:
        rets (pd.DataFrame): DataFrame with asset returns of the assets in the portfolio
    returns:
        pd.Series: portfolio weights
    """
    n_assets = len(rets.columns)
    weights = np.array([1/n_assets] * n_assets)
    return pd.Series(index=rets.columns,data=weights)

def get_mvp(rets: pd.DataFrame, min_vol_threshold=1e-6):
    """ find the minimum variance portofilio compisition
    Args:
        rets (pd.DataFrame): DataFrame with asset returns 
        min_vol_threshold : minimum threshold to eliminate constant prices
    returns:
        portfolio weights
    """
    # expected_returns=get_expected_returns_CAPM(returns_and_mkt,risk_free_rate).dropna()
    filtered_rets = rets.dropna(axis=1)

    # Compute standard deviations and filter assets below the threshold
    vol = filtered_rets.std()
    valid_assets = vol[vol > min_vol_threshold].index
    filtered_rets = filtered_rets[valid_assets]

    if filtered_rets.empty:
        print("No assets meet the volatility criteria.")
        return None

    # Calculate covariance matrix for the remaining assets
    cov_matrix = filtered_rets.cov()
    cov_matrix = cov_matrix.dropna(how='all').dropna(axis=1)
    n_assets = len(cov_matrix)
    initial_weights = np.array([1/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    gmv_result = minimize(
        minimize_volatility,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    gmv_weights = pd.Series(index=cov_matrix.columns, data=gmv_result.x)
    return gmv_weights


def get_mvp(rets: pd.DataFrame, cov_matrix, min_vol_threshold=1e-6)->pd.Series:
    """ find the minimum variance portofilio compisition
    Args:
        rets (pd.DataFrame): DataFrame with asset returns 
        min_vol_threshold : minimum threshold to eliminate constant prices
    returns:
        pd.Series: portfolio weights
    """
    # expected_returns=get_expected_returns_CAPM(returns_and_mkt,risk_free_rate).dropna()
    filtered_rets = rets.dropna(axis=1)

    # Compute standard deviations and filter assets below the threshold
    vol = filtered_rets.std()
    valid_assets = vol[vol > min_vol_threshold].index
    filtered_rets = filtered_rets[valid_assets]

    if filtered_rets.empty:
        print("No assets meet the volatility criteria.")
        return None

    # cov_matrix = cov_matrix.dropna(how='all').dropna(axis=1)
    n_assets = len(cov_matrix)
    initial_weights = np.array([1/n_assets] * n_assets)
    bounds = tuple((0, 1) for _ in range(n_assets))
    gmv_result = minimize(
        minimize_volatility,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    gmv_weights = pd.Series(index=cov_matrix.columns, data=gmv_result.x)
    return gmv_weights


def get_max_sharpe_portfolio(
    rets: pd.DataFrame,
    risk_free_rate: float = 0.0,
    days_in_sample: int = 365,
    min_vol_threshold: float = 1e-6,
    expected_returns: Optional[pd.Series] = None,
    cov_matrix: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Calculate the portfolio weights that maximize the Sharpe ratio.

    Args:
        rets (pd.DataFrame): DataFrame with asset returns.
        risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
        days_in_sample (int): Number of days to annualize returns and volatility.
        min_vol_threshold (float): Minimum volatility threshold to filter assets with constant prices.
        expected_returns (Optional[pd.Series]): Expected returns for assets. If None, use historical means.
        cov_matrix (Optional[pd.DataFrame]): Covariance matrix of asset returns. If None, compute from rets.

    returns:
        pd.Series: Portfolio weights, indexed by asset names, or None if optimization fails.
    """
    # Filter out assets with missing data or low volatility
    # filtered_rets = rets.drop('market',axis=1)
    filtered_rets = rets

    vol = filtered_rets.std()
    valid_assets = vol[vol > min_vol_threshold].index
    filtered_rets = filtered_rets[valid_assets]

    # Compute expected returns and cov matrix if not provided
    if expected_returns is None:
        # arithmetic annualization for markowitz
        expected_returns = filtered_rets.mean() * days_in_sample
    else:
        expected_returns = expected_returns.loc[valid_assets]

    if cov_matrix is None:
        cov_matrix = filtered_rets.cov()
    else:
        cov_matrix = cov_matrix.loc[valid_assets, valid_assets]

    # cond_number = np.linalg.cond(cov_matrix.values)
    # print(f"Covariance matrix condition number: {cond_number}")

    n_assets = len(valid_assets)
    # initial_weights = np.array([1/n_assets] * n_assets)
    initial_weights = np.random.dirichlet(np.ones(n_assets), size=1)[0]

    bounds = tuple((0, 1) for _ in range(n_assets))

    # Define negative Sharpe ratio for minimization
    def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate, days_in_sample):
        port_return = np.sum(expected_returns * weights) * days_in_sample
        port_vol = np.sqrt(np.dot(weights.T, np.dot(
            cov_matrix * days_in_sample, weights)))
        if port_vol == 0:
            return np.inf  # Avoid division by zero
        return -(port_return - risk_free_rate) / port_vol

    # Optimize
    result = minimize(
        negative_sharpe_ratio,
        initial_weights,
        args=(expected_returns, cov_matrix, risk_free_rate, days_in_sample),
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    )

    if not result.success:
        print(f"Optimization failed: {result.message}")
        return None
    # if not result.success:
    #     print(f"\nOptimization failed: {result.message}")
    #     print(f"Expected returns:\n{expected_returns}")
    #     print(f"Covariance matrix:\n{cov_matrix}")
    #     print(f"Initial weights:\n{initial_weights}")
    #     print(f"Volatilities:\n{vol}")
    #     print(f"Filtered returns shape: {filtered_rets.shape}")
    #     print(f"All-zero returns columns:\n{filtered_rets.columns[filtered_rets.std() < 1e-6].tolist()}")
    #     return None

    weights = pd.Series(data=result.x, index=valid_assets)
    return weights

# def backtest(rets, weight_func=get_mvp, rebalancing='Y', risk_free_rate=0.0, days_in_sample=365):
#     """Backtest the GMV portfolio strategy with specified rebalancing frequency.

#     Args:
#         returns_and_mkt (pd.DataFrame): DataFrame with asset returns and market returns.
#         rebalancing (str): Rebalancing frequency ('M' for monthly, 'Y' for yearly, etc.).
#         risk_free_rate (float): Risk-free rate for Sharpe ratio (default: 0.0).
#         days_in_sample: number of days to annualize

#     Returns:
#         dict: Portfolio performance metrics and time series.
#     """

#     # Group data by rebalancing frequency
#     rets_copy = rets.copy()
#     periods = rets_copy.groupby(pd.Grouper(freq=rebalancing))

#     portfolio_returns = pd.Series(dtype=float)
#     for period_start, period_data in periods:
#         if period_data.empty:
#             print('period_start is empty', period_start)
#             continue

#         weights = weight_func(period_data)
#         period_returns = period_data[weights.index]
#         period_portfolio_returns = (period_returns * weights).sum(axis=1)
#         portfolio_returns = pd.concat(
#             [portfolio_returns, period_portfolio_returns])

#     portfolio_returns = portfolio_returns.sort_index()
#     cumulative_returns = (1 + portfolio_returns).cumprod()

#     # Performance metrics
#     total_return = cumulative_returns.iloc[-1] - 1
#     days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
#     annualized_return = (1 + total_return) ** (days_in_sample / days) - 1
#     annualized_volatility = portfolio_returns.std() * np.sqrt(days_in_sample)
#     sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

#     return {
#         'portfolio_returns': portfolio_returns,
#         'cumulative_returns': cumulative_returns,
#         'total_return': total_return,
#         'annualized_return': annualized_return,
#         'annualized_volatility': annualized_volatility,
#         'sharpe_ratio': sharpe_ratio
#     }


def backtest(
    rets: pd.DataFrame,
    weight_func: Callable,
    rebalancing: str = 'Y',
    risk_free_rate: float = 0.0,
    days_in_sample: int = 365,
    expected_returns_func: Optional[Callable] = None,
    covariance_func: Optional[Callable] = None,
    expected_returns_kwargs: dict = {},
    covariance_kwargs: dict = {},
    weight_func_kwargs: dict = {}
) -> Dict[str, Any]:

    rets_copy = rets.copy()
    periods = rets_copy.groupby(pd.Grouper(freq=rebalancing))
    portfolio_returns = pd.Series(dtype=float)

    for period_start, period_data in periods:
        if period_data.empty:
            continue

        # Drop assets that are not traded yet (NaNs in this period) and with no vol (price is constant)
        period_data = period_data.dropna(axis=1, how='any')
        period_vol = period_data.std()
        valid_asset = period_vol[period_vol > 1e-6].index
        period_data = period_data[valid_asset]
        # invalid_asset = period_vol[period_vol<1e-6].index
        # print(invalid_asset)
        if period_data.shape[1] == 0:
            print(f"No assets traded in period {period_start}")
            continue

        try:
            # Compute expected returns and covariance if functions are provided
            expected_returns = None
            if expected_returns_func:
                expected_returns = expected_returns_func(
                    period_data, **expected_returns_kwargs)

            cov_matrix = None
            if covariance_func:
                cov_matrix = covariance_func(period_data, **covariance_kwargs)
            else:
                cov_matrix = period_data.cov()

            # Compute weights
            candidate_args = {
                "rets": period_data,
                "expected_returns": expected_returns,
                "cov_matrix": cov_matrix,
                "risk_free_rate": risk_free_rate,
                "days_in_sample": days_in_sample,
                **weight_func_kwargs
            }

            # valid args for weight_func
            sig = inspect.signature(weight_func)
            valid_args = {k: v for k, v in candidate_args.items()
                          if k in sig.parameters}
            weights = weight_func(**valid_args)

            if weights is None or isinstance(weights, (float, int)):
                print(f"Invalid weights returned at {period_start}")
                continue

            weights = pd.Series(weights).reindex(period_data.columns).fillna(0)
            period_portfolio_returns = (period_data * weights).sum(axis=1)
            portfolio_returns = pd.concat(
                [portfolio_returns, period_portfolio_returns])

        except Exception as e:
            print(f"Error at {period_start}: {e}")
            continue

    if portfolio_returns.empty:
        raise ValueError("No valid portfolio returns computed.")

    portfolio_returns = portfolio_returns.sort_index()
    cumulative_returns = (1 + portfolio_returns).cumprod()

    total_return = cumulative_returns.iloc[-1] - 1
    days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
    annualized_return = (1 + total_return) ** (days_in_sample / days) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(days_in_sample)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio
    }


def plot_backtest_vs_eq(
    rets: pd.DataFrame,
    market: pd.Series,
    weight_func: Callable = get_mvp,
    rebalancing: str = 'YE',
    risk_free_rate: float = 0.0,
    days_in_sample: int = 365,
    **weight_func_kwargs: Any
) -> None:
    """
    Plot the log-scale performance of a backtested portfolio against an equal-weighted market portfolio.

    Args:
        rets (pd.DataFrame): DataFrame with asset returns.
        market (pd.Series): Market returns.
        weight_func (Callable): Function to compute portfolio weights (default: get_mvp).
        rebalancing (str): Rebalancing frequency ('M', 'Y', etc.).
        risk_free_rate (float): Risk-free rate.
        days_in_sample (int): Number of days to annualize.
        **weight_func_kwargs: Additional arguments for weight_func.

    """
    backtest_results = backtest(
        rets,
        weight_func=weight_func,
        rebalancing=rebalancing,
        risk_free_rate=risk_free_rate,
        days_in_sample=days_in_sample,
        **weight_func_kwargs
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=np.log((market + 1).cumprod()),
                 label='Market (Equal-Weighted)')
    sns.lineplot(
        data=np.log((backtest_results['portfolio_returns'] + 1).cumprod()),
        label=weight_func.__name__
    )
    plt.title('Performance in Log Scale')
    plt.xlabel('Date')
    plt.ylabel('Log price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
