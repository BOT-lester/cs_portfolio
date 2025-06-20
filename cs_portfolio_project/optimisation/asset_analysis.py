import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from cs_portfolio_project.optimisation.portfolio import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union, Callable
from cs_portfolio_project.constant import freq_to_days
from pathlib import Path
from sklearn import covariance, cluster, manifold
from matplotlib.collections import LineCollection

project_root = Path(__file__).resolve().parents[2]


def calculate_market_returns(returns: pd.DataFrame) -> pd.Series:
    """ calulates the "market returns", returns of the equal weight portofilio of all assets in the returns df.
    """
    market_returns = pd.Series(index=returns.index, dtype=float)
    for date in returns.index:
        # Get available assets (non-NaN returns) for this date
        available_returns = returns.loc[date].dropna()
        n_assets = len(available_returns)
        if n_assets > 0:
            # Equal weights for available assets
            weights = np.array([1/n_assets] * n_assets)
            # Calculate market return for this date
            market_returns.loc[date] = (available_returns * weights).sum()
        else:
            market_returns.loc[date] = np.nan  # No data available
    return market_returns


def calculate_market_returns2(returns: pd.DataFrame) -> pd.Series:
    """Calculates market returns as the equal-weighted portfolio return of all assets."""
    return returns.mean(axis=1, skipna=True)


def get_returns_function(skin_type_csv: str) -> pd.DataFrame:
    """ get the returns for a csv file with daily prices of assets
    """
    prices = pd.read_csv(skin_type_csv)
    prices = prices.set_index(prices['date'])
    prices.drop('date', axis=1, inplace=True)
    return prices.pct_change().dropna(how='all')


def get_market_returns(skin_type_csv: str):
    """ calulates the "market returns", returns of the equal weight portofilio of all assets in returns df.
    Input: 
        skin_type_csv: name of csv file with all assets prices.
    output: market returns daily
    """
    returns = get_returns_function(skin_type_csv)
    mkt_return = calculate_market_returns(returns)
    return returns, mkt_return


def mdd(x):
    """computes max draw down"""
    wealth = (x+1).cumprod()
    cummax = wealth.cummax()
    drawdown = wealth/cummax - 1
    return drawdown.min()



def asset_information(returns, days_in_sample=365):
    """ compute total returns, avg returns (daily and ann), std (ann), max draw down """
    df = pd.DataFrame(index=returns.columns)
    df['total returns'] = (returns+1).cumprod().iloc[-1]
    df['average returns'] = returns.mean()
    df['average returns (ann)'] = (
        returns.mean()+1)**days_in_sample-1  # geometric annualization
    df['std (ann)'] = returns.std()*np.sqrt(days_in_sample)
    df['max draw down'] = returns.apply(mdd)
    df['max return'] = returns.max()
    df['min return'] = returns.min()
    return df


def get_data_and_market(rets: pd.DataFrame, market_rets):
    rets['market'] = market_rets
    return rets


def plot_correlation_matrix(correlation_matrix, figure_size=(25, 15)):

    plt.figure(figsize=figure_size)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('Correlation Matrix of Item Returns')
    plt.tight_layout()
    plt.show()


def plot_volatilty(returns, rolling_window):
    rolling_volatility = returns.rolling(window=rolling_window).std()
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_volatility, lw=1.5)
    plt.title(f"Rolling Volatility ({rolling_window}-day)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid()
    plt.show()


def plot_players_and_asset(players_pctchange, asset_returns, log=0):
    players_cum_log = (players_pctchange + 1).cumprod()
    asset_cum_log = (asset_returns + 1).cumprod()
    if log == 1:
        players_cum_log = np.log(players_cum_log)
        asset_cum_log = np.log(asset_cum_log)
    # Create a subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add trace for players' cumulative log returns (left y-axis)
    fig.add_trace(
        go.Scatter(x=players_pctchange.index, y=players_cum_log,
                   name="Players Cumulative Returns", line=dict(color="blue")),
        secondary_y=False,
    )

    # Add trace for asset cumulative log returns (right y-axis)
    fig.add_trace(
        go.Scatter(x=asset_returns.index, y=asset_cum_log,
                   name="Asset Cumulative Returns", line=dict(color="red")),
        secondary_y=True,
    )

    # Update layout with titles and axis labels
    fig.update_layout(
        title_text="Cumulative Returns: Players vs Asset",
        xaxis_title="Date",
        legend=dict(x=0.01, y=0.99),  # Position legend inside plot
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Players Returns", secondary_y=False)
    fig.update_yaxes(title_text="Asset Returns", secondary_y=True)

    # Make the plot interactive and show it
    fig.show()


def get_ewma_cov_matrix(returns, lambda_=0.94):
    """
    Compute the EWMA covariance matrix with weights w_t = lambda^(T-t) / sum(lambda^(T-t)).

    Args:
        returns (pd.DataFrame): Returns data, columns are assets.
        lambda_ (float): Decay factor for EWMA (0 < lambda_ < 1, default 0.94).
        days_in_sample (int): Number of days for annualization.

    Returns:
        pd.DataFrame:  EWMA covariance matrix.
    """
    returns = returns.drop(columns=['market'], errors='ignore')
    span = 2 / (1 - lambda_) - 1
    cov_ewma = returns.ewm(span=span, adjust=True).cov().iloc[-len(returns):]
    cov_matrix = cov_ewma.loc[returns.index[-1]]
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    # cov_matrix *= days_in_sample
    return cov_matrix


def compute_graphical_network(rets: pd.DataFrame, min_samples: int = 5) -> dict:
    """
    Compute clusters, partial correlations, and 2D embedding for asset returns.

    Args:
        rets (pd.DataFrame): DataFrame with asset returns, columns as assets, index as dates.
        min_samples (int): Minimum number of non-NaN observations required per asset.

    Returns:
        dict: Dictionary containing clustering labels, partial correlations, embedding, etc.
    """
    X = rets.drop(columns=['market'], errors='ignore').copy()

    valid_assets = X.columns[X.notna().sum() >= min_samples]
    if len(valid_assets) < 2:
        raise ValueError("Insufficient valid assets with enough non-NaN returns.")
    X = X[valid_assets].dropna(how='all')

    names = np.array(X.columns)
    print('Number of assets examined:', X.shape[1])

    # Standardize returns
    X_std = X / X.std(axis=0)
    X_std = X_std.fillna(0)

    # Graphical Lasso
    edge_model = covariance.GraphicalLassoCV(max_iter=1000)
    edge_model.fit(X_std)

    # Affinity propagation clustering
    _, labels = cluster.affinity_propagation(edge_model.covariance_)

    # MDS embedding
    embedding = manifold.MDS(n_components=2, random_state=0).fit_transform(X_std.T).T

    # Partial correlations
    prec = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(prec))
    partial_corr = prec * d
    partial_corr = partial_corr * d[:, np.newaxis]

    return {
        'names': names,
        'labels': labels,
        'partial_correlations': partial_corr,
        'embedding': embedding,
        'scaling_d': d,
        'start': X.index[0],
        'end': X.index[-1]
    }


def plot_graphical_network(rets: pd.DataFrame, min_samples: int = 5):
    """
    Plot the graphical network using the results from compute_graphical_network().

    Args:
        rets (pd.DataFrame): DataFrame with asset returns, columns as assets, index as dates.
        min_samples (int): Minimum number of non-NaN observations required per asset.
    """
    results=compute_graphical_network(rets, min_samples)

    names = results['names']
    labels = results['labels']
    partial_correlations = results['partial_correlations']
    embedding = results['embedding']
    d = results['scaling_d']
    start = results['start']
    end = results['end']
    for i in range(max(labels)+ 1):
        print(f'Cluster {i + 1}: {", ".join(names[labels == i])}')
        
    val_max = np.abs(partial_correlations[np.triu_indices_from(partial_correlations, k=1)]).max()
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    n_labels = labels.max()
    color_list = sns.color_palette("tab10", n_labels + 1)
    my_colors = [color_list[i] for i in labels]

    fig = plt.figure(figsize=(12, 6), facecolor='w')
    plt.clf()
    ax = plt.axes([0.05, 0.05, 0.9, 0.9])
    plt.axis('off')

    start_idx, end_idx = np.where(non_zero)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(
        segments,
        zorder=0,
        cmap=plt.cm.hot_r,
        norm=plt.Normalize(0, 0.7 * val_max)
    )
    lc.set_array(values)
    lc.set_linewidths(np.minimum(15 * values, 5))
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label('Partial Correlation Strength')

    plt.scatter(embedding[0], embedding[1], s=500 * d ** 2, c=my_colors, alpha=0.7)
    for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        ha = 'left' if this_dx > 0 else 'right'
        va = 'bottom' if this_dy > 0 else 'top'
        xo = 0.002 if this_dx > 0 else -0.002
        yo = 0.002 if this_dy > 0 else -0.002
        plt.text(
            x + xo, y + yo, name, size=8,
            ha=ha, va=va,
            bbox=dict(facecolor='w', edgecolor=color_list[label], alpha=0.6)
        )

    plt.xlim(embedding[0].min() - 0.15 * np.ptp(embedding[0]),
             embedding[0].max() + 0.10 * np.ptp(embedding[0]))
    plt.ylim(embedding[1].min() - 0.03 * np.ptp(embedding[1]),
             embedding[1].max() + 0.03 * np.ptp(embedding[1]))
    title = f'Graphical Network Analysis of Assets (Returns, {start.strftime("%m/%d/%Y")} to {end.strftime("%m/%d/%Y")})'
    plt.title(title, fontsize=12, pad=10)
    plt.show()


class AssetAnalysis:
    """ Class to manipulate assets
    """
    COV_METHODS = {
        'standard': lambda returns: returns.drop(columns=['market'], errors='ignore').cov() ,
        'ewma': lambda returns, lambda_=0.94: get_ewma_cov_matrix(returns, lambda_)
    }

    def __init__(self, csv_or_df: str | pd.DataFrame, risk_free_rate: float = 0.0, resample_size='D', cov_method: Union[str, Callable] = 'standard', lambda_: float = 0.94):
        """
        Initializes the class by loading data and computing necessary values.

        Args:
            csv_or_df: Path to CSV file or pandas DataFrame.
            risk_free_rate: Risk-free rate for return calculations.
            resample_size: Frequency for resampling ('D' for daily, etc.).
            cov_method: Method for covariance calculation ('standard', 'ewma', or custom function).
            lambda_: Decay factor for EWMA covariance (if applicable, default 0.94).
        """
        self.days_in_sample = freq_to_days[resample_size]
        self.risk_free_rate = risk_free_rate
        self.lambda_ = lambda_
        if isinstance(csv_or_df, str):
            # If a string (file path) is provided, load the CSV file
            self.data = pd.read_csv(
                csv_or_df, index_col='date', parse_dates=True)
            self.data.index = self.data.index.tz_localize(
                None).dropna(how='all')

        elif isinstance(csv_or_df, pd.DataFrame):
            # If a DataFrame is provided, use it directly
            self.data = csv_or_df.copy()
            self.data.index = self.data.index.tz_localize(
                None).dropna(how='all')

        else:
            raise ValueError(
                "Data must be a file path (str) or a pandas DataFrame.")

        # self.data = pd.read_csv(csv_file, index_col='date', parse_dates=True)
        if resample_size == 'D':
            self.returns = self.data.pct_change().dropna(how='all')

        else:
            self.returns = ((self.data.pct_change().dropna(
                how='all')+1).resample(resample_size).prod()-1).replace(0, np.nan)

        self.marketret = calculate_market_returns(self.returns)
        self.information = asset_information(
            self.returns, days_in_sample=self.days_in_sample)
        self.rets_and_market = get_data_and_market(
            self.returns.copy(), self.marketret)
        self.log_rets_and_market = np.log(1+self.rets_and_market)
        # self.cov_matrix = self.rets_and_market.drop(columns='market').cov()
        if isinstance(cov_method, str):
            if cov_method not in self.COV_METHODS:
                raise ValueError(
                    f"cov_method must be one of {list(self.COV_METHODS.keys())} or a callable")
            cov_func = self.COV_METHODS[cov_method]
            if cov_method == 'ewma':
                self.cov_matrix = cov_func(
                    self.rets_and_market, lambda_=self.lambda_)
            else:
                self.cov_matrix = cov_func(
                    self.rets_and_market)
        elif callable(cov_method):
            # Custom function: must return a pd.DataFrame
            self.cov_matrix = cov_method(
                self.rets_and_market)
            if not isinstance(self.cov_matrix, pd.DataFrame):
                raise ValueError(
                    "Custom cov_method must return a pandas DataFrame")
        else:
            raise ValueError("cov_method must be a string or callable")

        self.correlation_matrix = self.returns.corr()
        self.mwp = get_minimum_var_portfolio(
            self.returns, self.marketret, self.cov_matrix, risk_free_rate, self.days_in_sample)
        self.distance_matrix = 1 - self.correlation_matrix
        csv_path_players = project_root / 'data' / 'processed' / 'other' / 'players_monthly.csv'

        self.players = pd.read_csv(csv_path_players, index_col='Month')
        self.players.index =  pd.to_datetime(self.players.index)

    def plot_corr_matrix(self, figure_size=(25, 15)):
        """Plots the correlation matrix only when explicitly called."""
        plot_correlation_matrix(self.correlation_matrix, figure_size)

    def alpha_and_beta(self):
        """ get alpha and beta using CAPM"""
        return get_alpha_and_beta(self.returns, self.marketret)

    def plot_eff_frontiere(self, n_points, risk_free_rate=None):
        """Plot the efficient frontier.

        Args:
            n_points (int): Number of points to plot on the frontier.
            risk_free_rate (float, optional): Risk-free rate. Defaults to the value set at initialization.
        """
        effective_risk_free_rate = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        plot_efficient_frontier_2(
            self.returns,self.marketret,self.cov_matrix, n_points, effective_risk_free_rate)

    def plot_corr_with_market(self):
        correlation = self.rets_and_market.corr(
        )['market'].drop('market').sort_values()
        sns.barplot(y=correlation, x=correlation.index)
        plt.xticks(rotation='vertical')

    def plot_alpha_and_beta(self):
        """Plot alpha and beta coefficients."""
        plot_alpha_beta(self.alpha_and_beta())

    def plot_market_vol(self, rolling_window=15):
        "plot volatility of market"
        plot_volatilty(self.rets_and_market['market'], rolling_window)

    def plot_price(self, name: str | list[str], logscale: bool = False, start_date: str = None):
        """
        Plot the price or cumulative return of assets and/or the market.

        Args:
            name (str | list[str]): Name(s) of the asset(s) to plot. Can include 'market'.
            logscale (bool): Whether to use a logarithmic scale for the y-axis.
            start_date (str): Optional. A date string (e.g., '2022-01-01') to start plotting from.
        """
        fig = go.Figure()

        if isinstance(name, str):
            name = [name]

        data = self.data.copy()
        rets_market = self.rets_and_market.copy()

        if start_date is not None:
            data = data.loc[data.index >= start_date]
            rets_market = rets_market.loc[rets_market.index >= start_date]

        for asset in name:
            if asset in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[asset],
                                        mode='lines+markers', name=asset))
            elif asset == 'market':
                cumulative_returns = (1 + rets_market['market']).cumprod()
                fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                                        mode='lines+markers', name='Market'))
            else:
                print(f"Warning: {asset} not found in data")

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode="x unified",
            yaxis_type='log' if logscale else 'linear'
        )

        fig.show()

    def plot_returns_distribution(self, bins, exclude_0=True,log_rets=False):
        """Interactive plot of the distribution of all asset returns.
        Args:
            bins: Number of bins for the distribution plot.
            exclude_0: 1 to exclude the returns equal to 0
        """
        
        all_returns = self.returns.melt(value_name="Returns")[
            "Returns"].dropna()
        if log_rets:
            all_returns = np.log(1+ all_returns)
        if exclude_0:
            all_returns = all_returns[all_returns != 0]

        #  histogram
        fig = px.histogram(
            x=all_returns,
            title="Distribution of All Asset Returns",
            nbins=bins,
            histnorm='probability density',  # normalize for density
        )
        fig.update_traces(
            hoverinfo='x+y',
        )
        fig.update_layout(
            xaxis_title='Returns',
            yaxis_title='Density',
            hovermode='x unified',
            bargap=0.1,
        )

        fig.show()
        print(f'mean={all_returns.mean()}, std={all_returns.std()}')
        print(
            f'skewness={all_returns.skew()}, kurtosis={all_returns.kurtosis()}')

    def Kmeans_PCA_plot(self, n_clusters:int):
        # Reduce dimensionality with PCA
        clusters = kmean_clustering(n_clusters, self.distance_matrix)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.distance_matrix)
        df_plot = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
        df_plot['Cluster'] = clusters
        df_plot['Asset'] = self.returns.columns
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster',
                        data=df_plot, palette='tab10', s=100)
        for i, asset in enumerate(df_plot['Asset']):
            plt.text(df_plot.iloc[i, 0], df_plot.iloc[i,
                     1], asset, fontsize=9, ha='right')
        plt.title("K-Means Clustering of Assets (PCA Projection)")
        plt.show()

    def plot_players(self, asset='market', log=0):
        plot_players_and_asset(
            self.players['Change_pct'], self.rets_and_market[asset], log)
        
    def plot_graphical_network_assets(self,start_date:str):
        plot_graphical_network(self.returns[start_date:])
