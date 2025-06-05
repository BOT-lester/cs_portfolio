from numpy.linalg import inv
import pandas as pd
import numpy as np
from cs_portfolio_project.optimisation.asset_analysis import portfolio_return, portfolio_volatility, minimize_volatility
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_bl_delta(rets, rf=0):
    return (rets.mean()-rf)/rets.var()


def implied_returns(delta, sigma, w):
    """
    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
    delta: Risk Aversion Coefficient (scalar)
    sigma: Variance-Covariance Matrix (N x N) as DataFrame
        w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of Returns as Series
    """
    ir = delta * \
        sigma.dot(w).squeeze()  # series from a 1 column df
    ir.name = 'Implied Returns'
    return ir


# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)), index=p.index, columns=p.index)


def create_bl_views(rets, outperforming_assets, outperformance_values, weights=None):
    """Create P and Q matrices for Black-Litterman views."""
    # default are equal weights 
    if weights is None:
        weights = pd.Series(1/len(rets.columns), index=rets.columns)

    # errors
    if len(outperforming_assets) != len(outperformance_values):
        raise ValueError(
            " mismatch in number of outperforming assets and outperformance values")
    if not all(asset in rets.columns for asset in outperforming_assets):
        raise ValueError("Outperforming assets must be in returns columns")


    n_views = len(outperforming_assets)
    Q = pd.Series(outperformance_values, index=[
                  f"View_{i+1}" for i in range(n_views)], name='Q')
    P = pd.DataFrame(0.0, index=Q.index, columns=rets.columns)

    # Calculate total weight of "other" assets
    other_assets = [
        asset for asset in rets.columns if asset not in outperforming_assets]
    total_other_weight = weights[other_assets].sum()


    for i, (asset, outperformance) in enumerate(zip(outperforming_assets, outperformance_values)):
        P.loc[f"View_{i+1}", asset] = 1.0
        for other_asset in other_assets:
            if total_other_weight > 0:
                P.loc[f"View_{i+1}", other_asset] = - \
                    weights[other_asset] / total_other_weight
            else:
                P.loc[f"View_{i+1}", other_asset] = -1.0 / len(other_assets)

    return P, Q


def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=0.02):
    """Computes the posterior expected returns based on the Black-Litterman model."""
    # indices for consistence
    assets = sigma_prior.index
    views = p.index
    w_prior = w_prior.reindex(assets, fill_value=0)
    p = p.reindex(index=views, columns=assets, fill_value=0)
    q = q.reindex(views, fill_value=0)

    # input errors
    if w_prior.isna().any() or sigma_prior.isna().any().any() or p.isna().any().any() or q.isna().any():
        raise ValueError("Inputs must not contain NaN")
    if len(w_prior) != sigma_prior.shape[0] or p.shape[1] != len(w_prior):
        raise ValueError("Matrix dimensions misaligned")

    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p) # K x K matrix
    omega = pd.DataFrame(omega, index=views, columns=views)

    pi = implied_returns(delta, sigma_prior, w_prior)
    pi = pi.reindex(assets)  # Ensure alignment

    sigma_prior_scaled = tau * sigma_prior

    term1 = sigma_prior_scaled.dot(p.T)  # N x K
    term1 = pd.DataFrame(term1, index=assets, columns=views)

    # K x K
    temp = p.dot(sigma_prior_scaled).dot(p.T) + omega
    term2 = np.linalg.inv(temp)  
    term2 = pd.DataFrame(term2, index=views, columns=views)


    p_dot_pi = p.dot(pi)  # K x 1, index=views
    term3 = q - p_dot_pi 
    term3 = pd.Series(term3, index=views)

    intermediate = term1.dot(term2)  # N x K
    mu_bl_adjustment = intermediate.dot(term3)  # N x 1
    mu_bl = pi + mu_bl_adjustment

    sigma_bl = (sigma_prior + sigma_prior_scaled -
                term1.dot(term2).dot(p).dot(sigma_prior_scaled))

    return mu_bl, sigma_bl


def get_bl_mu_and_sigma(rets: pd.DataFrame, outperforming_assets: list, outperformance_values: list, rf=0, weights=None, delta=None, tau=None, omega=None, market_rets=None):
    """Compute BL return (mu) and cov matrixe (sigma)
    Args:
        rets: returns of the assets
        outperforming_assets : list of the names of outperforming assets 
        outperformance_values: list of the % of outperformance (same order)
        weights: weight of the market (Default is eq weight)
        delta : risk aversion parameter (default )
        tau : uncertainty of prior (default 1/T)
    output:
        mu_bl, sigma_bl
    """
    if delta == None:
        delta = get_bl_delta(market_rets, rf)
    if tau == None:
        tau = 1/len(rets)
    if weights is None:
        weights = pd.Series(1/len(rets.columns), index=rets.columns)

    p, q = create_bl_views(rets, outperforming_assets,
                           outperformance_values, weights)
    sigma_prior = rets.cov()

    mu_bl, sigma_bl = bl(weights, sigma_prior, p, q,
                         omega, delta=2.5, tau=0.02)

    return mu_bl, sigma_bl

def get_bl_mu(data, **kwargs):
    mu, _ = get_bl_mu_and_sigma(data, **kwargs)
    return mu

def get_bl_sigma(data, **kwargs):
    _, sigma = get_bl_mu_and_sigma(data, **kwargs)
    return sigma

def get_bl_efficient_frontier(mu_bl, sigma_bl, n_points=50, risk_free_rate=0.0):
    """
    Compute the efficient frontier using Black-Litterman posterior returns and covariance.

    Args:
        mu_bl (pd.Series): Posterior expected returns from BL model, indexed by assets.
        sigma_bl (pd.DataFrame): Posterior covariance matrix from BL model, indexed by assets.
        n_points (int): Number of points on the frontier.
        risk_free_rate (float): Risk-free rate for return adjustment.

    Returns:
        tuple: (eff_returns, eff_volatilities) for plotting the efficient frontier.
    """
    # Ensure inputs are aligned
    assets = mu_bl.index
    if not (sigma_bl.index.equals(assets) and sigma_bl.columns.equals(assets)):
        raise ValueError("sigma_bl indices must match mu_bl index")

    n_assets = len(assets)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # sum weights = 1
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # No shorting
    initial_weights = np.array([1/n_assets] * n_assets)

    # Range of target returns (from min to max possible return)
    min_return = mu_bl.min()
    max_return = mu_bl.max()
    target_returns = np.linspace(min_return, max_return, n_points)

    efficient_frontier = []
    for target_ret in target_returns:
        constraints.append(
            {'type': 'eq', 'fun': lambda w: portfolio_return(w, mu_bl) - target_ret})
        result = minimize(
            minimize_volatility,
            initial_weights,
            args=(sigma_bl,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            volatility = portfolio_volatility(result.x, sigma_bl)
            efficient_frontier.append((target_ret, volatility))
        constraints.pop()  # Remove the return constraint for next iteration

    # Convert to arrays for plotting
    eff_returns, eff_volatilities = zip(
        *efficient_frontier) if efficient_frontier else ([], [])
    return eff_returns, eff_volatilities


def plot_bl_efficient_frontier(mu_bl, sigma_bl, market_vol=None, market_ret=None, n_points=50, risk_free_rate=0.0):
    """
    Plot the efficient frontier using BL posterior returns and covariance.

    Args:
        mu_bl (pd.Series): Posterior expected returns from BL model.
        sigma_bl (pd.DataFrame): Posterior covariance matrix from BL model.
        market_vol (float, optional): Market portfolio volatility (for reference).
        market_ret (float, optional): Market portfolio return (for reference).
        n_points (int): Number of points on the frontier.
        risk_free_rate (float): Risk-free rate.
    """
    eff_returns, eff_volatilities = get_bl_efficient_frontier(
        mu_bl, sigma_bl, n_points, risk_free_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(eff_volatilities, eff_returns,
             'b-', label='BL Efficient Frontier')

    if market_vol is not None and market_ret is not None:
        plt.scatter([market_vol], [market_ret], color='red',
                    marker='o', label='Market Portfolio')
        plt.text(market_vol, market_ret, 'Market',
                 fontsize=7, ha='right', va='bottom')

    for asset in mu_bl.index:
        asset_vol = np.sqrt(sigma_bl.loc[asset, asset])
        asset_ret = mu_bl[asset]
        plt.scatter([asset_vol], [asset_ret], label=asset)
        plt.text(asset_vol, asset_ret, asset,
                 fontsize=7, ha='left', va='bottom')

    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Black-Litterman Efficient Frontier with Assets and Market')

    plt.grid(True)
    plt.show()
