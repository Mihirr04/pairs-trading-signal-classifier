# Pairs Trading Signal Classifier

A pairs trading research pipeline that identifies statistically valid stock pairs via cointegration testing, labels mean-reversion events using leakage-free expanding-window statistics, and trains a logistic regression classifier on engineered spread features to distinguish true convergence signals from noise.

Built with Python, scikit-learn, statsmodels, and yfinance.

---

## Project Overview

Pairs trading is a market-neutral strategy that exploits temporary divergences between two historically related stocks. When the spread between them widens beyond its typical range, a trader shorts the over-priced stock and longs the under-priced one, betting the spread will revert.

The hard part is distinguishing **real mean-reversion opportunities** from **noise** — spreads that diverge and stay diverged. This project frames that distinction as a binary classification problem.

---

## Pipeline

1. **Data** — Download 5 years of daily closing prices (2019-2024) for candidate pairs via `yfinance`.
2. **Pair Validation** — Engle-Granger cointegration test on each pair to filter for statistically valid candidates.
3. **Spreads** — Linear regression to derive the residual spread for each pair.
4. **Rolling Half-Life** — Compute mean-reversion half-life on a 60-day rolling window per event.
5. **Labeling** — Expanding-window mean/std to identify divergence events; label as `signal` if the spread reverts within 20 days, else `noise`.
6. **Features** — For each spike event, extract `z-score`, `velocity`, `rolling half-life`, and `days outside band`.
7. **Classifier** — Chronological 80/20 split, standard scaling, logistic regression with manual class weights and tuned probability threshold (0.65).
8. **Visualization** — Four diagnostic plots covering regression fits, decision boundaries, time-series labels, and global performance.

---

## Results

### Cointegration

Twelve candidate pairs were tested. **1 of 12** passed the Engle-Granger test at p < 0.05:

| Pair | Correlation | Coint. p-value | Status |
|---|---|---|---|
| V/MA | 0.98 | 0.008 | ✓ Cointegrated |
| KO/PEP | 0.95 | 0.071 | ✗ Not cointegrated |
| AAPL/MSFT | 0.97 | 0.367 | ✗ Not cointegrated |
| HD/LOW | 0.97 | 0.090 | ✗ Not cointegrated |
| XOM/CVX | 0.97 | 0.176 | ✗ Not cointegrated |
| *(7 more)* | | | ✗ Not cointegrated |

**Key finding:** High correlation does not imply cointegration. Most well-correlated pairs in this universe fail the stationarity test, particularly over a period that includes COVID-era regime shifts.

### Classifier Performance

Trained on 1,683 spike events, evaluated on 421 out-of-sample events (chronological split).

| Metric | Value |
|---|---|
| Overall accuracy | 0.96 |
| Signal precision | 0.00 |
| Signal recall | 0.00 |
| Class balance (test) | 405 noise / 16 signal |

The headline accuracy is **misleading**: signal events are rare (≈4% of the test set), and the model defaults to predicting the majority class. See **Limitations** below.

### Diagnostic Plots

All four plots are auto-generated to `results/`:

- `01_regression_scatter.png` — Linear regression fits per pair
- `02_decision_boundaries.png` — Per-pair logistic decision boundaries in (z-score, velocity) space
- `03_spreads_with_signals.png` — Spreads over time with signal/noise labels
- `04_model_performance.png` — Confusion matrix, feature importance, P/R, probability distribution

---

## Repo Structure

pairs-trading-signal-classifier/
├── src/
│   ├── data.py         # yfinance downloading
│   ├── pairs.py        # pair definitions, correlation, cointegration
│   ├── features.py     # spreads, rolling half-life, feature matrix
│   ├── labels.py       # expanding-window labeling
│   ├── model.py        # train/test split, classifier
│   └── plots.py        # diagnostic visualizations
├── legacy/
│   └── pairs_trading.py  # original single-file implementation
├── results/            # auto-generated plots
├── run.py              # main entry point
├── requirements.txt
└── README.md

---

## How to Run

```bash
git clone https://github.com/Mihirr04/pairs-trading-signal-classifier.git
cd pairs-trading-signal-classifier

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python run.py
```

Outputs print to console; plots save to `results/`.

---

## Methodology Notes

A few design choices worth flagging:

- **Chronological split.** Random k-fold leaks future information into training. Chronological 80/20 simulates real out-of-sample deployment.
- **Expanding-window labeling.** Each event's "is this outside the band" judgment uses only data available *up to that day*, preventing lookahead bias in the labels themselves.
- **Per-event rolling half-life.** Half-life is computed on a 60-day window ending at each event, not as a single constant per pair. This makes it a genuinely informative feature that varies with the spread's recent dynamics.
- **Manual class weighting.** `class_weight='balanced'` overcompensates on imbalanced data. Manual `sqrt(noise/signal)` weighting performed empirically better in informal testing.
- **Tuned threshold.** A 0.5 probability cutoff fires too eagerly given the class imbalance; 0.65 trades recall for precision.

---

## Limitations

This is a research project, not a production trading system. Known limitations:

1. **Linear model ceiling.** Logistic regression on 4 hand-engineered features is fundamentally limited. The spread dynamics that distinguish real mean-reversion from noise are likely nonlinear, regime-dependent, and richer than four scalars can capture.
2. **Extreme class imbalance.** With expanding-window labeling, real signals are ~4-7% of all spike events. The classifier degenerates toward the majority class and the headline accuracy becomes uninformative. F1 on the signal class is the metric that matters, and it is currently near zero.
3. **Cointegration is fragile.** Only 1 of 12 candidate pairs passes the Engle-Granger test on 2019-2024 data. The pair universe is too narrow to support a robust strategy, and even cointegrated pairs can decouple structurally (mergers, business model shifts).
4. **No backtest.** Classification accuracy is not equivalent to trading P&L. Without a backtest that accounts for entry/exit rules, holding periods, and transaction costs, the model's economic value is unmeasured.
5. **Static feature set.** Features are engineered, not learned. Real practitioners use richer representations: order-flow imbalance, volatility regimes, sector spreads, and macro factors.
6. **Single asset class, single market.** US equities only. Pairs trading is more commonly applied to ETFs, commodities, or cross-market pairs where structural relationships are stronger.

---

## Future Work

Specific extensions, in rough order of expected impact:

- **Backtest engine.** Convert classification predictions into a simulated strategy with entry/exit rules, holding-period limits, and a SPY benchmark. Report Sharpe, max drawdown, and turnover.
- **Expanded pair universe.** Test 50-100 candidate pairs (including ETF pairs) and filter via cointegration to build a more diversified portfolio.
- **Nonlinear classifier.** Gradient boosted trees (XGBoost / LightGBM) as a direct comparison to the linear baseline.
- **Dynamic hedge ratio.** Replace static OLS with a Kalman filter to allow the hedge ratio to evolve over time.
- **OU-process modeling.** Model the spread as an Ornstein-Uhlenbeck process and use the analytical mean-reversion speed as both a feature and a trading signal.
- **Regime detection.** Hidden Markov Model or rolling structural break tests to gate trading during regimes where cointegration breaks down.

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `statsmodels` · `yfinance` · `matplotlib`

---

## Author

Mihir Shinde — [GitHub](https://github.com/Mihirr04)