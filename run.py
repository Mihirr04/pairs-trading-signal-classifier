"""
Pairs Trading Signal Classifier — main entry point.

Run with: python run.py

Pipeline:
1. Download price data
2. Compute correlations + cointegration tests
3. Compute spreads and rolling half-lives
4. Build labels
5. Build feature matrix
6. Train logistic regression classifier
7. Generate all plots
"""

from src.data import download_prices
from src.pairs import (
    PAIRS,
    compute_correlations,
    test_cointegration,
    filter_cointegrated_pairs,
)
from src.features import (
    compute_spreads,
    compute_rolling_halflives,
    build_feature_matrix,
)
from src.labels import build_labels
from src.model import train_classifier
from src.plots import (
    plot_regression_scatter,
    plot_decision_boundaries,
    plot_spreads_with_signals,
    plot_model_performance,
)


# =============================================================================
# CONFIG
# =============================================================================

START      = '2019-01-01'
END        = '2024-01-01'
MIN_WINDOW = 60
HL_WINDOW  = 60
THRESHOLD  = 0.65


# =============================================================================
# 1. DOWNLOAD DATA
# =============================================================================

print("=" * 60)
print("1. Downloading price data")
print("=" * 60)

tickers  = list({t for pair in PAIRS for t in pair})
raw_data = download_prices(tickers, START, END)
print(f"Shape: {raw_data.shape}")


# =============================================================================
# 2. CORRELATIONS + COINTEGRATION
# =============================================================================

print("\n" + "=" * 60)
print("2. Pair correlations and cointegration tests")
print("=" * 60)

correlations  = compute_correlations(raw_data, PAIRS)
coint_results = test_cointegration(raw_data, PAIRS, significance=0.05)

print(f"\n{'Pair':<12} {'Corr':>7} {'Coint p':>10} {'Status':>20}")
print("-" * 52)
for s1, s2 in PAIRS:
    pair_name = f"{s1}/{s2}"
    corr      = correlations[pair_name]
    pval      = coint_results[pair_name]['pvalue']
    status    = "✓ cointegrated" if coint_results[pair_name]['cointegrated'] else "✗ not cointegrated"
    print(f"{pair_name:<12} {corr:>7.4f} {pval:>10.4f} {status:>20}")

valid_pairs = filter_cointegrated_pairs(PAIRS, coint_results)
print(f"\n{len(valid_pairs)}/{len(PAIRS)} pairs passed cointegration test.")


# =============================================================================
# 3. SPREADS + ROLLING HALF-LIVES
# =============================================================================

print("\n" + "=" * 60)
print("3. Computing spreads and rolling half-lives")
print("=" * 60)

spreads = compute_spreads(raw_data, PAIRS)
print(f"Spreads computed for {len(spreads)} pairs.")

print("Computing rolling half-lives (takes ~30s)...")
rolling_hl = compute_rolling_halflives(spreads, window=HL_WINDOW)
for pair_name, hl in rolling_hl.items():
    median_hl = hl.replace(999.0, float('nan')).median()
    print(f"  {pair_name}: median rolling half-life = {median_hl:.1f} days")


# =============================================================================
# 4. LABELS
# =============================================================================

print("\n" + "=" * 60)
print("4. Building labels (expanding-window)")
print("=" * 60)

labels = build_labels(spreads, min_window=MIN_WINDOW)
for pair_name, label_series in labels.items():
    n_sig   = label_series.sum()
    n_noise = (label_series == 0).sum()
    print(f"  {pair_name}: signals={n_sig}, noise={n_noise}")


# =============================================================================
# 5. FEATURE MATRIX
# =============================================================================

print("\n" + "=" * 60)
print("5. Building feature matrix")
print("=" * 60)

X, y, pair_names = build_feature_matrix(spreads, rolling_hl, labels,
                                        min_window=MIN_WINDOW)
print(f"Total spike events: {len(y)}")
print(f"Real signals: {y.sum()}  |  Noise: {(y == 0).sum()}")


# =============================================================================
# 6. TRAIN CLASSIFIER
# =============================================================================

print("\n" + "=" * 60)
print("6. Training logistic regression classifier")
print("=" * 60)

train_result = train_classifier(X, y, test_size=0.2, threshold=THRESHOLD)


# =============================================================================
# 7. PLOTS
# =============================================================================

print("\n" + "=" * 60)
print("7. Generating plots")
print("=" * 60)

plot_regression_scatter(raw_data, PAIRS)
plot_decision_boundaries(spreads, rolling_hl, labels, PAIRS,
                          min_window=MIN_WINDOW, threshold=THRESHOLD)
plot_spreads_with_signals(spreads, labels, PAIRS)
plot_model_performance(train_result)

print("\n" + "=" * 60)
print("Done. Results saved to results/")
print("=" * 60)