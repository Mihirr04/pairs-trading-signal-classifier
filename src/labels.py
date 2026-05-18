"""
Labeling module.

For each day a spread is outside its 2-sigma band, label it:
    1 = signal (spread converges back toward mean within 20 days)
    0 = noise  (spread does not converge)

Uses expanding-window mean/std to avoid lookahead bias.
"""

import pandas as pd


def build_labels(spreads, min_window=60, future_window=20, convergence_factor=0.5):
    """
    Generate binary labels for each spread event.

    Parameters
    ----------
    spreads : dict
        Pair name -> pd.Series of spread values.
    min_window : int
        Minimum days of history before labeling begins.
    future_window : int
        How many days forward to check for convergence.
    convergence_factor : float
        Spread is "converged" if it falls within
        (mean - factor * std, mean + factor * std).

    Returns
    -------
    dict
        Pair name -> pd.Series of 0/1 labels aligned with the spread.
    """
    labels = {}

    for pair_name, spread in spreads.items():
        pair_labels = pd.Series(0, index=spread.index)

        for i in range(min_window, len(spread) - future_window):
            history = spread.iloc[:i]
            mean    = history.mean()
            std     = history.std()

            if std == 0:
                continue

            upper = mean + 2 * std
            lower = mean - 2 * std
            val   = spread.iloc[i]

            if val > upper or val < lower:
                future    = spread.iloc[i + 1 : i + 1 + future_window]
                converged = ((future - mean).abs() < convergence_factor * std).any()
                pair_labels.iloc[i] = 1 if converged else 0

        labels[pair_name] = pair_labels

    return labels