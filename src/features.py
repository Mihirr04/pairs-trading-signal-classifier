"""
Feature engineering module.

Builds:
- Spreads via linear regression on each pair
- Rolling half-life of mean reversion (per-event feature)
- Feature matrix for the classifier:
    [zscore, velocity, rolling_hl, days_outside]
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_spreads(data, pairs):
    """
    For each pair, fit y = beta * x and compute the residual spread.

    Parameters
    ----------
    data : pd.DataFrame
        Price data, columns are tickers.
    pairs : list of tuple
        List of (ticker1, ticker2) pairs.

    Returns
    -------
    dict
        {'KO/PEP': pd.Series of spread values, ...}
    """
    spreads = {}
    for s1, s2 in pairs:
        X = data[s1].values.reshape(-1, 1)
        y = data[s2].values

        model = LinearRegression()
        model.fit(X, y)
        spread = y - model.predict(X)

        spreads[f"{s1}/{s2}"] = pd.Series(spread, index=data.index)
    return spreads


def rolling_halflife(spread_series, window=60):
    """
    Compute rolling half-life of mean reversion.

    For each day i, fit delta_spread = alpha + beta * spread_lag
    on the prior `window` days. Half-life = -ln(2) / beta.

    Parameters
    ----------
    spread_series : pd.Series
        Spread values indexed by date.
    window : int
        Rolling window size in days.

    Returns
    -------
    pd.Series
        Half-life values aligned with spread_series index.
    """
    hl_values = pd.Series(np.nan, index=spread_series.index)

    for i in range(window, len(spread_series)):
        window_data = spread_series.iloc[i - window : i]
        lag         = window_data.shift(1).dropna()
        delta       = window_data.diff().dropna()
        lag         = lag.iloc[:len(delta)]

        if len(lag) < 10:
            continue

        X_hl = lag.values.reshape(-1, 1)
        y_hl = delta.values

        lr   = LinearRegression()
        lr.fit(X_hl, y_hl)
        beta = lr.coef_[0]

        hl_values.iloc[i] = -np.log(2) / beta if beta < 0 else 999.0

    hl_values = hl_values.bfill().fillna(999.0)
    return hl_values


def compute_rolling_halflives(spreads, window=60):
    """Apply rolling_halflife to every pair's spread."""
    return {
        pair_name: rolling_halflife(spread, window=window)
        for pair_name, spread in spreads.items()
    }


def build_feature_matrix(spreads, rolling_hl, labels, min_window=60):
    """
    Build the feature matrix for the classifier.

    For each day a spread is outside its 2-sigma band, extract:
        - zscore        : (val - mean) / std
        - velocity      : val - val_prev
        - rolling_hl    : half-life from the rolling window
        - days_outside  : consecutive days the spread has been outside the band

    Parameters
    ----------
    spreads : dict
        Pair name -> pd.Series of spread values.
    rolling_hl : dict
        Pair name -> pd.Series of rolling half-life values.
    labels : dict
        Pair name -> pd.Series of 0/1 labels.
    min_window : int
        Minimum days of history before generating features.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix indexed by date.
    y : pd.Series
        Labels indexed by date.
    pair_names : list
        Pair name for each row (for diagnostics).
    """
    all_features   = []
    all_labels     = []
    all_dates      = []
    all_pair_names = []

    for pair_name, spread in spreads.items():
        hl_series = rolling_hl[pair_name]

        for i in range(min_window + 1, len(spread) - 20):
            history = spread.iloc[:i]
            mean    = history.mean()
            std     = history.std()

            if std == 0:
                continue

            val   = spread.iloc[i]
            upper = mean + 2 * std
            lower = mean - 2 * std

            if val > upper or val < lower:
                velocity = val - spread.iloc[i - 1]
                zscore   = (val - mean) / std
                hl       = hl_series.iloc[i]

                # Count consecutive prior days outside the band
                days_outside = 0
                for j in range(i - 1, max(i - 21, min_window - 1), -1):
                    pv = spread.iloc[j]
                    ph = spread.iloc[:j]
                    if len(ph) < 2:
                        break
                    pm = ph.mean()
                    ps = ph.std()
                    if ps == 0:
                        break
                    if pv > pm + 2 * ps or pv < pm - 2 * ps:
                        days_outside += 1
                    else:
                        break

                all_features.append([zscore, velocity, hl, days_outside])
                all_labels.append(labels[pair_name].iloc[i])
                all_dates.append(spread.index[i])
                all_pair_names.append(pair_name)

    X = pd.DataFrame(
        all_features,
        columns=['zscore', 'velocity', 'rolling_hl', 'days_outside'],
        index=all_dates,
    )
    y = pd.Series(all_labels, index=all_dates)
    return X, y, all_pair_names