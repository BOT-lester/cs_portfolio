import pandas as pd
import numpy as np
from cs_portfolio_project.config import config


class CovEstimator:
    @staticmethod
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

    @staticmethod
    def get_constant_corr_covmatrix(returns):
        """
        Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
        """
        rhos = returns.corr()
        n = rhos.shape[0]
        rho_bar = (rhos.values.sum()-n)/(n*(n-1))
        ccor = np.full_like(rhos, rho_bar)
        np.fill_diagonal(ccor, 1.)
        sd = returns.std()
        ccov = ccor * np.outer(sd, sd)
    #     mh.corr2cov(ccor, sd)
        return pd.DataFrame(ccov, index=returns.columns, columns=returns.columns)

    @staticmethod
    def get_shrunk_covariance_matrix(returns: pd.DataFrame, shrinkage_matrix: pd.DataFrame = None, lambda_=0.5):
        """
        Compute the shrunk covariance matrix, lambda = 0 for shrunk matrix, lambda = 1 for sample covariance matrix.

        Args:
            returns (pd.DataFrame): Returns data, columns are assets.
            shrinkage_matrix: shrinkage matrix (ex: Elton/Gruber Constant Correlation matrix, default is identity matrix)
            lambda_ (float): shrinkage factor (0 < lambda_ < 1, default 0.5).

        Returns:
            pd.DataFrame:  shrunk covariance matrix.
        """
        n = len(returns.columns)
        if shrinkage_matrix is None:
            shrinkage_matrix = np.identity(n)

        returns = returns.drop(columns=['market'], errors='ignore')

        cov = returns.cov()
        return lambda_*cov + (1-lambda_)*shrinkage_matrix
    
    @staticmethod
    def get_shrunk_covariance_matrix_identity(returns: pd.DataFrame,lambda_=0.5): 
        return CovEstimator.get_shrunk_covariance_matrix(returns=returns, shrinkage_matrix= None, lambda_=lambda_)
    
    @staticmethod
    def get_shrunk_covariance_matrix_constant_corr(returns: pd.DataFrame, lambda_=0.5): 
        return CovEstimator.get_shrunk_covariance_matrix(returns=returns, shrinkage_matrix= CovEstimator.get_constant_corr_covmatrix(returns), lambda_=lambda_)



class ExpectedReturns:

    @staticmethod
    def get_expected_returns_CAPM(rets, market_rets, risk_free_rate=None):
        from cs_portfolio_project.optimisation.portfolio import get_alpha_and_beta
        if risk_free_rate==None:
            risk_free_rate=config.RISK_FREE_RATE
            
        expected_market_return = market_rets.mean()
        capm_results = get_alpha_and_beta(rets, market_rets).set_index('Asset')

        expected_returns = pd.Series({
            asset: risk_free_rate + capm_results.loc[asset, 'Beta'] * (
                expected_market_return - risk_free_rate) + capm_results.loc[asset, 'Alpha']
            for asset in capm_results.index
        })
        return expected_returns
    
    @staticmethod
    def get_expected_returns_historic_EWMA(returns, lambda_=0.94):
        """
        Compute expected returns using the Exponentially Weighted Moving Average (EWMA).

        Args:
            returns (pd.DataFrame): Asset returns, each column is an asset.
            lambda_ (float): Decay factor for EWMA (default 0.94).

        Returns:
            pd.Series: EWMA expected returns (latest value).
        """
        returns = returns.drop(columns=['market'], errors='ignore')
        span = 2 / (1 - lambda_) - 1
        returns_EWMA = returns.ewm(span=span, adjust=True).mean()

        # cov_matrix *= days_in_sample
        return returns_EWMA.iloc[-1]
    
    