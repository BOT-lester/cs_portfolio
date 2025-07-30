# app.py
import streamlit as st
from cs_portfolio_project.optimisation.asset_analysis import *
from cs_portfolio_project.optimisation.portfolio import *
from cs_portfolio_project.optimisation.montecarlo import *
from cs_portfolio_project.config import config
from cs_portfolio_project.optimisation.black_litterman import *
from cs_portfolio_project.optimisation.estimator_functions import *

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime

import os
st.title("CS2 Skins Portfolio Analysis Web App")

#  Sidebar
st.sidebar.header("Data Selection")
data_type = st.sidebar.selectbox("Choose data type", ["cases2", "agents","cases_and_agents_mkt"])
data_path = f"data/processed/D/{data_type}.csv"

risk_free_rate = st.sidebar.number_input("Risk-free rate", value=0.0)
resample_size = st.sidebar.selectbox("Resample size", ["D", "W","B", "M",'Y'], index=0)
cov_method = st.sidebar.selectbox("Covariance method", ["standard", "ewma", "shrunk_id", "shrunk_constant_corr"], index=0)
er_method = st.sidebar.selectbox("Expected returns method", ["capm", "standard", "historic_ewma"], index=0)

lambda_ = st.sidebar.slider("Lambda (for EWMA/shrunk)", min_value=0.0, max_value=1.0, value=0.94)



if "aa" not in st.session_state:
    st.session_state.aa = None

if "additional_asset_names" not in st.session_state:
    st.session_state.additional_asset_names = []
if "additional_asset_names_start_end_data" not in st.session_state:
    st.session_state.additional_asset_names_start_end_data = ["", ""]

# = st.sidebar.checkbox("Use expected returns function")

add_assets_button = st.sidebar.checkbox("### Add additional non CS assets")
if add_assets_button:
    options = list(config.ASSET_LABEL_TO_TICKER.keys())
    selected_labels = st.sidebar.multiselect("Choose assets to add", options)
    selected_tickers = [config.ASSET_LABEL_TO_TICKER[label] for label in selected_labels]
    # Store selected assets


    # Sidebar: Select start and end date
    start_date = st.sidebar.date_input("Start date for additional assets", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End date for additional assets", value=datetime.today())

# Save date selection
    if selected_tickers :
        st.session_state.additional_asset_names = selected_tickers
        st.session_state.additional_asset_names_start_end_data = [
            str(start_date), str(end_date)
        ]
    else:
        st.sidebar.write('Please select additional assets')
else:
    # Set to None to be cleanly ignored by AssetAnalysis if no assets selected
    st.session_state.additional_asset_names = None
    st.session_state.additional_asset_names_start_end_data = None



# create the AssetAnalysis object
if st.sidebar.button("Run Analysis"):
    st.session_state.aa = AssetAnalysis(
        data_path,
        risk_free_rate=risk_free_rate,
        resample_size=resample_size,
        cov_method=cov_method,
        er_method=er_method,
        lambda_=lambda_,
        additional_asset_names=st.session_state.additional_asset_names,
        additional_asset_names_start_end_data=st.session_state.additional_asset_names_start_end_data
    
    )
    st.success("Data loaded and analysis object created!")

if st.session_state.aa:
    plot_choice = st.sidebar.selectbox(
        "Choose a plot to display",
        [   "information",
            "Correlation Matrix",
            "Efficient Frontier",
            "Market ACF/PACF",
            "Market Decomposition",
            "Correlation With Market",
            "Alpha and Beta (CAPM)",
            "Market Volatility",
            "Asset Price Plot",
            "Returns Distribution",
            "KMeans PCA Clustering",
            "Players vs Asset",
            "Graphical Network of Assets",
            "Skins Quantity Portfolio Plot",
            "Backtest vs Equal Weight",
            "monte carlo simulation",
            "Black Litterman"
        ]
    )
    aa = st.session_state.aa
    marketret = aa.marketret

    if plot_choice == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        plt.figure()
        aa.plot_corr_matrix()
        st.pyplot(plt.gcf())

    if plot_choice == "information":
        st.subheader("information")
        st.dataframe(aa.information)
        st.subheader("market information")
        st.dataframe(asset_information(pd.DataFrame(aa.marketret),aa.days_in_sample),hide_index=True)

    elif plot_choice == "Efficient Frontier":
        st.subheader("Efficient Frontier")
        n_points = st.sidebar.slider("Number of frontier points", 10, 200, 50)
        plt.figure()
        aa.plot_eff_frontiere(n_points=n_points, risk_free_rate=risk_free_rate)
        st.pyplot(plt.gcf())

    elif plot_choice == "Market ACF/PACF":
        data_type = st.sidebar.selectbox("Data type", ["price", "returns"])
        lags = st.sidebar.slider("Lags", 5, 100, 20)
        st.subheader(f"Market ACF/PACF ({data_type})")
        plt.figure()
        aa.plot_market_ACF_PACF(data_type=data_type, lags=lags)
        st.pyplot(plt.gcf())

    elif plot_choice == "Market Decomposition":
        data_type = st.sidebar.selectbox("Data type", ["price", "returns"])
        model = st.sidebar.selectbox("Decomposition model", ["additive", "multiplicative"])
        scale = st.sidebar.slider("Bar scale factor", 1, 100, 5)
        plt.figure()
        aa.plot_market_decompose(data_type=data_type, model=model, bar_scaling_factor=scale)
        st.pyplot(plt.gcf())

    elif plot_choice == "Correlation With Market":
        plt.figure()
        aa.plot_corr_with_market()
        st.pyplot(plt.gcf())

    elif plot_choice == "Alpha and Beta":
        st.subheader("Alpha and Beta")
        plt.figure()
        aa.plot_alpha_and_beta()
        st.pyplot(plt.gcf())

    elif plot_choice == "Market Volatility":
        rolling = st.sidebar.slider("Rolling window", 5, 50, 15)
        st.subheader("Market Volatility")
        plt.figure()
        aa.plot_market_vol(rolling_window=rolling)
        st.pyplot(plt.gcf())

    elif plot_choice == "Asset Price Plot":
        all_assets = ["market"] + list(aa.data.columns)
        selected_assets = st.multiselect("Select asset(s)", all_assets, default="market")
        logscale = st.checkbox("Logarithmic scale")
        start_date = st.text_input("Start date (YYYY-MM-DD)", value="")
        fig = aa.plot_price(name=selected_assets, logscale=logscale, start_date=start_date or None)
        st.plotly_chart(fig,use_container_width=True)

    elif plot_choice == "Returns Distribution":
        bins = st.sidebar.slider("Bins", 10, 200, 50)
        exclude_0 = st.sidebar.checkbox("Exclude zero returns", value=True)
        log_rets = st.sidebar.checkbox("Log returns", value=False)
        fig, mean, std, skew, kurt = aa.plot_returns_dist(
            bins=bins, exclude_0=exclude_0, log_rets=log_rets)

        st.plotly_chart(fig, use_container_width=True)

        st.write(f"**Mean:** {mean:.4f}")
        st.write(f"**Std Dev:** {std:.4f}")
        st.write(f"**Skewness:** {skew:.4f}")
        st.write(f"**Kurtosis:** {kurt:.4f}")

    elif plot_choice == "KMeans PCA Clustering":
        clusters = st.sidebar.slider("Number of clusters", 2, 10, 3)
        plt.figure()
        aa.Kmeans_PCA_plot(n_clusters=clusters)
        st.pyplot(plt.gcf())

    elif plot_choice == "Players vs Asset":
        asset = st.selectbox("Select asset", ["market"] + list(aa.returns.columns))
        log = st.checkbox("Logarithmic returns")
        # plt.figure()
        fig = aa.plot_players(asset=asset, log=int(log))
        st.plotly_chart(fig,use_container_width=True)

    elif plot_choice == "Graphical Network of Assets":
        start_date = st.text_input("Start date (YYYY-MM-DD)", value="2022-01-01")
        plt.figure()
        aa.plot_graphical_network_assets(start_date=start_date)
        st.pyplot(plt.gcf())

    elif plot_choice == "Skins Quantity Portfolio Plot":
        portfolio_type = st.selectbox("Portfolio type", ["minimum variance", "equal weight", "max sharp"])
        funds = st.number_input("Available funds ($)", value=1000)
        st.write("Computing portfolio allocation...")
        weights = aa.get_weights_portfolio(portfolio_type=portfolio_type)[0]
        fig1, ax1 = plt.subplots()
        weights.plot.pie(autopct='%1.1f%%', ylabel='', ax=ax1)
        st.pyplot(fig1)
        plt.close(fig1)
        skins_quantity_plot,ptf_return_ann,ptf_volatility_ann,quantity=aa.find_skins_quantity_portfolio_and_plot(available_funds=funds, portfolio_type=portfolio_type)
        
        st.pyplot(skins_quantity_plot)
        st.write(f"Annualized expected returns: {ptf_return_ann:.4f}",)
        st.write(f"Annualized volatilty: {ptf_volatility_ann:.4f}")
        w_and_q_df = pd.concat([weights,quantity],axis=1)
        w_and_q_df.columns = ['weights','quantity']
        st.dataframe(w_and_q_df)

        
    # elif plot_choice == "Backtest vs Equal Weight":
    #     rebalancing = st.selectbox("Rebalancing frequency", ['M','2M', 'Q', 'YE'])
    #     risk_free = st.number_input("Risk-free rate (e.g., 0.02)", value=0.0)
    #     days = st.number_input("Annualization days", value=252)
    #     compare = st.checkbox("Compare with other assets")
    #     other_df = None

    #     if compare:
    #         selected = st.multiselect("Select assets to compare", aa.returns.columns)
    #         if selected:
    #             other_df = aa.returns[selected]

    #     fig = plot_backtest_vs_eq_webapp(
    #         aa.returns,
    #         aa.marketret,
    #         rebalancing=rebalancing,
    #         risk_free_rate=risk_free,
    #         days_in_sample=days,
    #         other_comparison_asset=other_df,
    #     )

    #     st.pyplot(fig)
    elif plot_choice == "Backtest vs Equal Weight":

        st.subheader("Backtest Parameters") 
        start_date_backtest = st.date_input("Start date for backtest", value=datetime(2013, 11, 21))
        rebalancing = st.selectbox("Rebalancing frequency", ['M', 'Q', '6M', 'YE'])
        risk_free = st.number_input("Risk-free rate", value=0.0)
        days = st.number_input("Annualization days", value=365)

        # ------------------------------------------
        # Weighting Function
        weight_func_name = st.selectbox("Weighting Function", ['get_mvp', 'get_max_sharpe_portfolio', 'get_equal_weight_pf'])

        if weight_func_name == 'get_mvp':
            weight_func = WeightFunctions.get_mvp
            min_vol_threshold = st.number_input("Min vol threshold", value=1e-6, format="%.6f")
            weight_func_kwargs = {"min_vol_threshold": min_vol_threshold}

        elif weight_func_name == 'get_max_sharpe_portfolio':
            weight_func = WeightFunctions.get_max_sharpe_portfolio
            min_vol_threshold = st.number_input("Min vol threshold", value=1e-6, format="%.6f")
            weight_func_kwargs = {
                "min_vol_threshold": min_vol_threshold,
                "risk_free_rate": risk_free,
                "days_in_sample": days
            }

        elif weight_func_name == 'get_equal_weight_pf':
            weight_func = WeightFunctions.get_equal_weight_pf
            weight_func_kwargs = {}

        # ------------------------------------------
        # Expected Returns Function
        use_exp_func = st.checkbox("Use expected returns function")
        if use_exp_func:
            exp_func_name = st.selectbox("Expected Returns Function", ['get_expected_returns_CAPM', 'default','Black Litterman'])
            is_bl = exp_func_name == 'Black Litterman'

            if exp_func_name == 'get_expected_returns_CAPM':
                expected_returns_func = ExpectedReturns.get_expected_returns_CAPM
                exp_kwargs = {
                    "risk_free_rate": risk_free,
                    "market_rets": aa.marketret
                }
            elif exp_func_name == 'Black Litterman':
                expected_returns_func = get_bl_mu
                asset_names = aa.returns.columns.tolist()
                skin_BL = {}
                valid_assets = []

                for asset in asset_names:
                    # weight = st.slider(f"Weight for {asset}", 0.0, 1.0, 0.0, 0.01)
                    BL_outperformance = st.number_input(f"outperformance of asset {asset} ", value=0.00000,step=0.0001, format="%.6f")
                    
                    if BL_outperformance != 0.0:
                        skin_BL[asset] = BL_outperformance
                exp_kwargs = {

                    "market_rets": marketret,
                    "outperforming_assets":list(skin_BL.keys()),
                    "outperformance_values":list(skin_BL.values())
                }
                
            else:
                expected_returns_func = None
                exp_kwargs = {}
        else:
            expected_returns_func = None
            exp_kwargs = {}
            is_bl= False

        # ------------------------------------------
        # Covariance Function
        if not is_bl:
            use_cov_func = st.checkbox("Use covariance function")
            if use_cov_func:
                cov_func_name = st.selectbox("Covariance Function", ['get_ewma_cov_matrix', 'get_shrunk_covariance_matrix_identity', 'get_shrunk_covariance_matrix_constant_corr','default',"Black Litterman"])
                if cov_func_name == 'get_ewma_cov_matrix':
                    covariance_func = CovEstimator.get_ewma_cov_matrix
                    lambda_ewma = st.slider("EWMA lambda", min_value=0.8, max_value=0.99, value=0.94)
                    cov_kwargs = {"lambda_": lambda_ewma}
                elif cov_func_name == 'get_shrunk_covariance_matrix_identity':
                    covariance_func =  CovEstimator.get_shrunk_covariance_matrix_identity
                    lambda_shrink = st.slider("Shrinkage lambda", min_value=0.0, max_value=1.0, value=0.5)
                    cov_kwargs = {"lambda_": lambda_shrink}
                elif cov_func_name == 'get_shrunk_covariance_matrix_constant_corr':
                    covariance_func =  CovEstimator.get_shrunk_covariance_matrix_constant_corr
                    lambda_shrink = st.slider("Shrinkage lambda", min_value=0.0, max_value=1.0, value=0.5)
                    cov_kwargs = {"lambda_": lambda_shrink}
                elif cov_func_name=="Black Litterman":
                    covariance_func =  get_bl_sigma

                    cov_kwargs = {
                        "market_rets": aa.marketret,
                        "outperforming_assets":list(skin_BL.keys()),
                        "outperformance_values":list(skin_BL.values())
                    }
                else:
                    covariance_func = None
                    cov_kwargs = {}
            else:
                covariance_func = None
                cov_kwargs = {}
        else:
            covariance_func = get_bl_sigma
            cov_kwargs = {
                "market_rets": aa.marketret,
                "outperforming_assets": list(skin_BL.keys()),
                "outperformance_values": list(skin_BL.values())
            }
        # ------------------------------------------
        # Plotting
        st.markdown("### Other Assets for comparison")
        options = list(config.ASSET_LABEL_TO_TICKER.keys())
        selected_labels_backtest = st.multiselect("Choose assets to add to backtest graph", options)
        selected_tickers_for_backtest = [config.ASSET_LABEL_TO_TICKER[label] for label in selected_labels_backtest]
        if selected_tickers_for_backtest == []:
            selected_tickers_for_backtest=None
        if st.button("Run Backtest"):
            fig = plot_backtest_vs_eq_webapp(
                rets=aa.returns[start_date_backtest:],
                market=marketret[start_date_backtest:],
                weight_func=weight_func,
                rebalancing=rebalancing,
                risk_free_rate=risk_free,
                days_in_sample=days,
                expected_returns_func=expected_returns_func,
                expected_returns_kwargs=exp_kwargs,
                covariance_func=covariance_func,
                covariance_kwargs=cov_kwargs,
                weight_func_kwargs=weight_func_kwargs,
                other_comparison_asset=selected_tickers_for_backtest
            )

            st.pyplot(fig)



    elif plot_choice == "monte carlo simulation":
        st.subheader("📊 Monte Carlo Simulation")
        # Risk-Free Rate
        risk_free = st.number_input("Risk-free rate", value=0.0)
        initial_portfolio_value = st.number_input("initial portfolio value", value=100.0)
        number_of_sims = st.slider("number of simulations",  10, 500, 100)
        sim_timeframe=st.number_input("number of periods to simulate", value=365)
        
        log = st.checkbox("use log returns")
        days = st.number_input("Annualization days", value=365)

        weight_func_name = st.selectbox("Weighting Function", ['get_mvp', 'get_max_sharpe_portfolio', 'get_equal_weight_pf','custom'])

        if weight_func_name == 'get_mvp':
            weight_func = WeightFunctions.get_mvp
            min_vol_threshold = st.number_input("Min vol threshold", value=1e-6, format="%.6f")
            weight_func_kwargs = {"min_vol_threshold": min_vol_threshold,"rets":aa.returns}

        elif weight_func_name == 'get_max_sharpe_portfolio':
            weight_func = WeightFunctions.get_max_sharpe_portfolio
            min_vol_threshold = st.number_input("Min vol threshold", value=1e-6, format="%.6f")
            weight_func_kwargs = {
                "min_vol_threshold": min_vol_threshold,
                "risk_free_rate": risk_free,
                "days_in_sample": days
            }

        elif weight_func_name == 'get_equal_weight_pf':
            weight_func = WeightFunctions.get_equal_weight_pf
            weight_func_kwargs = {}

        elif weight_func_name == 'custom':
            st.markdown("Set number of items.")
            
            asset_names = aa.returns.columns.tolist()
            skin_numbers = {}
            total_weight = 0

            for asset in asset_names:
                # weight = st.slider(f"Weight for {asset}", 0.0, 1.0, 0.0, 0.01)
                number = st.number_input(f"number of {asset}", value=0)
                
                skin_numbers[asset] = number
                # total_weight += weight

            skin_numbers_series = pd.Series(skin_numbers)
            weights_portfolio,value = find_portfolio_weights_and_value(skin_numbers_series.index,skin_numbers_series.to_list(),aa.data)

            st.write("Weights being used:")
            st.dataframe(weights_portfolio)
            weight_func = find_portfolio_weights
            weight_func_kwargs = {
                "skins": skin_numbers_series.index,
                "quantity": skin_numbers_series.to_list(),
                "asset_prices": aa.data
            }

        # ------------------------------------------
        # Expected Returns Function

        use_exp_func = st.checkbox("Use expected returns function")
        if use_exp_func:
            exp_func_name = st.selectbox("Expected Returns Function", ['CAPM expected returns', 'default','Black Litterman','EWMA historic returns'])
            is_bl = exp_func_name == 'Black Litterman'

            if exp_func_name == 'EWMA historic returns':
                expected_returns_func = ExpectedReturns.get_expected_returns_historic_EWMA
                lambda_ewma_rets = st.slider("EWMA lambda ER", min_value=0.100, max_value=0.999, value=0.940)
                exp_kwargs = {"lambda_": lambda_ewma_rets}
            elif exp_func_name == 'CAPM expected returns':
                expected_returns_func = ExpectedReturns.get_expected_returns_CAPM
                exp_kwargs = {
                    "risk_free_rate": risk_free,
                    "market_rets": aa.marketret
                }
            elif exp_func_name == 'Black Litterman':
                expected_returns_func = get_bl_mu
                asset_names = aa.returns.columns.tolist()
                skin_BL = {}
                valid_assets = []

                for asset in asset_names:
                    # weight = st.slider(f"Weight for {asset}", 0.0, 1.0, 0.0, 0.01)
                    BL_outperformance = st.number_input(f"outperformance of asset {asset} ", value=0.00000,step=0.0001, format="%.6f")
                    
                    if BL_outperformance != 0.0:
                        skin_BL[asset] = BL_outperformance
                exp_kwargs = {

                    "market_rets": marketret,
                    "outperforming_assets":list(skin_BL.keys()),
                    "outperformance_values":list(skin_BL.values())
                }
                
            else:
                expected_returns_func = None
                exp_kwargs = {}
        else:
            expected_returns_func = None
            exp_kwargs = {}
            is_bl= False

        # ------------------------------------------
        # Covariance Function
        if not is_bl:
            use_cov_func = st.checkbox("Use covariance function")
            if use_cov_func:
                cov_func_name = st.selectbox("Covariance Function", ['get_ewma_cov_matrix', 'get_shrunk_covariance_matrix_identity', 'get_shrunk_covariance_matrix_constant_corr','default',"Black Litterman"])
                if cov_func_name == 'get_ewma_cov_matrix':
                    covariance_func = CovEstimator.get_ewma_cov_matrix
                    lambda_ewma = st.slider("EWMA lambda COV", min_value=0.1, max_value=0.999, value=0.94)
                    cov_kwargs = {"lambda_": lambda_ewma}
                elif cov_func_name == 'get_shrunk_covariance_matrix_identity':
                    covariance_func =  CovEstimator.get_shrunk_covariance_matrix_identity
                    lambda_shrink = st.slider("Shrinkage lambda", min_value=0.0, max_value=1.0, value=0.5)
                    cov_kwargs = {"lambda_": lambda_shrink}
                elif cov_func_name == 'get_shrunk_covariance_matrix_constant_corr':
                    covariance_func =  CovEstimator.get_shrunk_covariance_matrix_constant_corr
                    lambda_shrink = st.slider("Shrinkage lambda", min_value=0.0, max_value=1.0, value=0.5)
                    cov_kwargs = {"lambda_": lambda_shrink}
                elif cov_func_name=="Black Litterman":
                    covariance_func =  get_bl_sigma

                    cov_kwargs = {
                        "market_rets": aa.marketret,
                        "outperforming_assets":list(skin_BL.keys()),
                        "outperformance_values":list(skin_BL.values())
                    }
                else:
                    covariance_func = None
                    cov_kwargs = {}
            else:
                covariance_func = None
                cov_kwargs = {}
        else:
            covariance_func = get_bl_sigma
            cov_kwargs = {
                "market_rets": aa.marketret,
                "outperforming_assets": list(skin_BL.keys()),
                "outperformance_values": list(skin_BL.values())
            }




        if st.button("Run Monte Carlo Simulation"):
            MC_sim,MC_sim_info,fig = simulate_portfolio_performance(
                rets=aa.returns,
                weight_func=weight_func,
                initial_portfolio_value=initial_portfolio_value,
                number_of_sims=number_of_sims,
                sim_timeframe=sim_timeframe,
                log=log,
                expected_returns_func=expected_returns_func,
                expected_returns_kwargs=exp_kwargs,
                covariance_func=covariance_func,
                covariance_kwargs=cov_kwargs,
                weight_func_kwargs=weight_func_kwargs
            )

            st.pyplot(fig)
            st.write(MC_sim_info)
    
    if plot_choice == "Black Litterman":

        st.markdown("Outperformance of a given asset compared to the market (ex: 0.1 for 10% outperformanfce)")
        
        asset_names = aa.returns.columns.tolist()
        skin_BL = {}

        for asset in asset_names:
            # weight = st.slider(f"Weight for {asset}", 0.0, 1.0, 0.0, 0.01)
            BL_outperformance = st.number_input(f"outperformance of asset {asset} ", value=0.00000,step=0.0001, format="%.6f")
            
            skin_BL[asset] = BL_outperformance
            # total_weight += weight
        plt.figure()
        mu_bl, sigma_bl=get_bl_mu_and_sigma(aa.returns,list(skin_BL.keys()),list(skin_BL.values()),market_rets=aa.marketret)
        plot_bl_efficient_frontier(mu_bl, sigma_bl)
        st.pyplot(plt.gcf())
else:
    st.info("Select your options and click 'Run Analysis' to begin.")