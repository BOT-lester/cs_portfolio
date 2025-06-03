
import pandas as pd
import numpy as np
import os
import argparse

def get_prices_daily(skin_type: str, resample_size='D'):
    folder_path = f'{skin_type}'

    condition_map = {
        "battle-scarred": "BS",
        "well-worn": "WW",
        "field-tested": "FT",
        "minimal-wear": "MW",
        "factory-new": "FN",
        "minimal_wear": "MW",
        "foil": "f",
        "holo-foil": "h_f",


    }

    price_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])

            # Clean base name
            base_name = filename.replace('_price_history.csv', '').replace(
                '_price.csv', '').replace('_case', '').replace("sticker_capsule", "SC")
            new_column_name = base_name  # Default name

            for cond, abbreviation in condition_map.items():
                if f"({cond})" in base_name:
                    new_column_name = base_name.replace(
                        f"_({cond})", "") + f"_{abbreviation}"
                    break  # Stop after finding the first match

            if new_column_name:
                price_data[new_column_name] = df.set_index('date')['price']

    if not price_data:
        raise ValueError("No valid price data found.")

    # Combine all prices into a single DataFrame
    all_prices = pd.concat(price_data, axis=1, join='outer')
    all_prices_daily = all_prices.resample(resample_size).mean()

    return all_prices_daily

# def smooth_prices(prices_df,same_time_starting, rol_window_size=60,n_std=1):
#     """ Smoothes the prices of iliquid items (remove large price spikes and interpolate missing values)
#     output: dataframe of smoothed prices
#     """
#     median_price = prices_df.rolling(rol_window_size, min_periods=1).median()
#     deviation = (prices_df - median_price).abs()

#     threshold = n_std * prices_df.rolling(rol_window_size, min_periods=1).std()
#     prices_df[deviation > threshold] = median_price  # Replace outliers with median
#     prices_df = prices_df.interpolate(method='linear')
#     if same_time_starting == 1:
#         prices_df = prices_df.bfill()
#     return prices_df


def adjust_spikes(series, spike_window=12, spike_deviation_threshold=0.2, spike_reversion_window=3):
    """
    Adjust price spikes in a sparse sales series (cain contain NaNs between sales).

    Parameters:
    - series: pandas Series with datetime index, containing prices and NaNs
    - window: number of past sales for baseline (default: 5)
    - deviation_threshold: max deviation from baseline to flag a spike (default: 0.2 or 20%)
    - reversion_window: number of future sales to check for reversion (default: 3)

    Returns:
    - pandas Series with spikes adjusted and NaNs forward-filled
    """
    sales_series = series.dropna()

    adjusted_sales = pd.Series(index=sales_series.index, dtype=float)
    for i in range(len(sales_series)):
        if i == 0:
            # First sale
            adjusted_sales.iloc[i] = sales_series.iloc[i]
            continue

        # Calculate baseline from previous 'window' sales
        past_sales = sales_series.iloc[max(0, i - spike_window):i]
        baseline = past_sales.median()
        current_price = sales_series.iloc[i]

        # Check for spike
        deviation = (current_price - baseline) / baseline
        if abs(deviation) > spike_deviation_threshold:
            # Check future sales for reversion
            future_sales = sales_series.iloc[i +
                                             1:i + 1 + spike_reversion_window]
            if not future_sales.empty:
                # If any future sale reverts to within threshold of baseline
                reversion = (future_sales - baseline).abs() / \
                    baseline < spike_deviation_threshold
                if reversion.any():
                    adjusted_sales.iloc[i] = baseline # Spike confirmed: replace with baseline
                else:
                    adjusted_sales.iloc[i] = current_price
            else:
                # No future sales to check: keep original price
                adjusted_sales.iloc[i] = current_price
        else:
            # Not a spike: keep original price
            adjusted_sales.iloc[i] = current_price

    #map adjusted sales back to original series
    adjusted_series = series.copy()
    adjusted_series.loc[adjusted_sales.index] = adjusted_sales
    # adjusted_series = adjusted_series.ffill()

    return adjusted_series


def smooth_and_fill_with_moving_average(series, smoothing_window=5, spike_window=12, spike_deviation_threshold=0.2, spike_reversion_window=3):
    """
    Smooth prices and fill missing values using a moving average based on sales.

    Parameters:
    - series: pandas Series with datetime index, containing prices and NaNs
    - smoothing_window: number of sales (not days) for the moving average (default: 5)

    Returns:
    - pandas Series with spikes adjusted, smoothed with a sales-based moving average, and NaNs filled
    """
    spike_adjusted_series = adjust_spikes(
        series, spike_window, spike_deviation_threshold, spike_reversion_window)

    #sales data (non-NaN values) and apply moving average
    sales_series = spike_adjusted_series.dropna()
    smoothed_sales = sales_series.rolling(
        window=smoothing_window, min_periods=1, center=True).mean()

    #Map smoothed sales back to the original series and interpolate to fill NaNs
    smoothed_series = spike_adjusted_series.copy()
    smoothed_series.loc[smoothed_sales.index] = smoothed_sales
    smoothed_series = smoothed_series.interpolate(method='time')

    return smoothed_series

# def filter_low_prices_until_threshold(df, threshold=0.08):

#     def filter_column(col):
#         # Find the index of the first value >= threshold
#         mask_above_threshold = col <= threshold
#         if mask_above_threshold.any():
#             first_under_idx = mask_above_threshold.idxmax()  # First False value
#             first_above_idx = (col.loc[col.index >=first_under_idx] >=threshold).idxmax() # First value abov threshold again
#             # Replace values with NaN before the first threshold crossing
#             col.loc[:first_above_idx] = np.nan
#         return col

#     # Apply the filtering to each column
#     filtered_df = df.apply(filter_column)
#     return filtered_df


def filter_low_prices_until_threshold(df, threshold=0.06, min_days_diff=15):
    """
    Replace prices below threshold with NaN until the first price exceeds it, 
    keeping all subsequent prices. Usefull for cases that had constant values
    around 0.03€ before beeing removed from the active drop pool.
    
    Filters so that we only have prices only when item is no longuer dropping
    because when cases are dropping their price is mostly constant.
    But if case price drops below thershold after being above, the price is
    kept.
    Parameters:
    - df: pandas DataFrame with datetime index and assets as columns
    - threshold: price threshold (default: 0.06 €)
    - min_days_diff: number of days of the price <= threshold to start fitler
    Returns:
    - pandas DataFrame with low prices filtered until threshold is crossed

    Exemple :
        [0.5, 0.2, 0.05, 0.05, 0.15, 0.04, 0.3, 0.7]
        => [Nan,Nan, Nan, Nan, 0.15, 0.07, 0.3, 0.7]
    """
    def filter_column(col):
        col_copy = col.copy()
        start_idx = col.index[0]  

        while True:  # continues if min_days_diff is below 15
            # Find the first value <= threshold from the current start point
            mask_under_threshold = col_copy.loc[col_copy.index >=
                                                start_idx] <= threshold
            if not mask_under_threshold.any():
                break  # No more values <= threshold, stop

            first_under_idx = mask_under_threshold.idxmax()  # First True value (<= threshold)

            # find first value >= threshold after first_under_idx
            mask_above_threshold = col_copy.loc[col_copy.index >=
                                                first_under_idx] >= threshold
            if not mask_above_threshold.any():
                break  # stop their are no more values ≥ threshold
            first_above_idx = mask_above_threshold.idxmax()  # First True value (≥ threshold)

            # if time_diff is > min_days_diff, apply filtering and stop
            # helps in case of high volatity when a case just realeased and the price can drop below
            # teh threshold evne if the case if still dropping
            time_diff = (first_above_idx - first_under_idx).days
            if time_diff > min_days_diff:
                col.loc[:first_above_idx] = np.nan
                break
            else:
                # Move the start point to just after first_above_idx and continue
                start_idx = col.index[col.index > first_above_idx][0] if col.index[col.index >
                                                                                   first_above_idx].size > 0 else None
                if start_idx is None:
                    break  

        return col

    filtered_df = df.apply(filter_column)
    return filtered_df


def save_prices(skin_type: str, smooth, remove_active_drop, name='', resample_size='D', smoothing_window=5, spike_window=12, spike_deviation_threshold=0.2, spike_reversion_window=3):
    """ Convert all skins prices from a given folder to a single csv file
    Input: 
        skin_type: name of the skin type folder
        smooth: 1 if prices need to be smoothed (for iliquid items)
        remove_active_drop: 1 if only prices after drop pool should be kept.4
    """

    raw_path = os.path.join("data", "raw", "market_prices", skin_type)
    prices = get_prices_daily(raw_path, resample_size)
    # sort by realse date
    first_sale_dates = prices.apply(lambda col: col.first_valid_index())
    sorted_columns = first_sale_dates.sort_values().index
    prices_sorted = prices[sorted_columns]
    prices_sorted.resample(resample_size).mean()
    if remove_active_drop:
        prices_sorted = filter_low_prices_until_threshold(prices_sorted)

    # apply smoothing for illiquid items
    if smooth:
        prices_sorted = prices_sorted.apply(lambda column: smooth_and_fill_with_moving_average(
            column, smoothing_window, spike_window, spike_deviation_threshold, spike_reversion_window))
    
    filename = f"{name.lower()}.csv" if name else f"{skin_type.lower()}.csv"
    output_path = os.path.join("data", "processed", resample_size, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prices_sorted.to_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skin_type", type=str, required=True)
    parser.add_argument("--smooth", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--remove_active_drop", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--resample_size", type=str, default="D")
    parser.add_argument("--smoothing_window", type=int, default=5)
    parser.add_argument("--spike_window", type=int, default=12)
    parser.add_argument("--spike_deviation_threshold", type=float, default=0.2)
    parser.add_argument("--spike_reversion_window", type=int, default=3)

    args = parser.parse_args()

    save_prices(
        skin_type=args.skin_type,
        smooth=args.smooth,
        remove_active_drop=args.remove_active_drop,
        name=args.name,
        resample_size=args.resample_size,
        smoothing_window=args.smoothing_window,
        spike_window=args.spike_window,
        spike_deviation_threshold=args.spike_deviation_threshold,
        spike_reversion_window=args.spike_reversion_window
    )