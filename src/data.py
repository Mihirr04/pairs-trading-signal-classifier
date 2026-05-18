"""
Data loading module.
Downloads daily closing prices from Yahoo Finance.
"""

import yfinance as yf


def download_prices(tickers, start, end):
    """
    Download daily closing prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Stock tickers, e.g. ['KO', 'PEP', 'AAPL'].
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame of closing prices, indexed by date, columns are tickers.
    """
    data = yf.download(tickers, start=start, end=end)['Close']
    data.sort_index(inplace=True)
    return data