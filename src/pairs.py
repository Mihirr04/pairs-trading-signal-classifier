"""
Pair definitions and statistical tests for pair validity.

Includes:
- The list of candidate pairs
- Pearson correlation of daily prices
- Engle-Granger cointegration test (new)
"""

from statsmodels.tsa.stattools import coint


PAIRS = [
    ('KO',   'PEP'),
    ('AAPL', 'MSFT'),
    ('AMD',  'NVDA'),
    ('V',    'MA'),
    ('NKE',  'LULU'),
    ('HD',   'LOW'),
    ('MCD',  'YUM'),
    ('XOM',  'CVX'),
    ('JPM',  'BAC'),
    ('JNJ',  'PFE'),
    ('DAL',  'UAL'),
    ('COST', 'WMT'),
]


def compute_correlations(data, pairs):
    """
    Pearson correlation of daily closing prices for each pair.

    Parameters
    ----------
    data : pd.DataFrame
        Price data, columns are tickers.
    pairs : list of tuple
        List of (ticker1, ticker2) pairs.

    Returns
    -------
    dict
        {'KO/PEP': 0.94, ...}
    """
    return {
        f"{s1}/{s2}": data[s1].corr(data[s2])
        for s1, s2 in pairs
    }


def test_cointegration(data, pairs, significance=0.05):
    """
    Engle-Granger cointegration test for each pair.

    A pair is cointegrated if their spread is stationary (mean-reverting).
    p < significance rejects the null of no cointegration.

    Parameters
    ----------
    data : pd.DataFrame
        Price data, columns are tickers.
    pairs : list of tuple
        List of (ticker1, ticker2) pairs.
    significance : float
        p-value threshold. Default 0.05 (95% confidence).

    Returns
    -------
    dict
        {'KO/PEP': {'pvalue': 0.012, 'cointegrated': True, 'test_stat': -3.45}, ...}
    """
    results = {}
    for s1, s2 in pairs:
        score, pvalue, _ = coint(data[s1].dropna(), data[s2].dropna())
        results[f"{s1}/{s2}"] = {
            'test_stat': score,
            'pvalue': pvalue,
            'cointegrated': pvalue < significance,
        }
    return results


def filter_cointegrated_pairs(pairs, coint_results):
    """Return only pairs that passed the cointegration test."""
    return [
        (s1, s2) for s1, s2 in pairs
        if coint_results[f"{s1}/{s2}"]['cointegrated']
    ]