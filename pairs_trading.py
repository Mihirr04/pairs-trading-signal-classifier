# =============================================================================
# PAIRS TRADING SIGNAL CLASSIFIER — IMPROVED VERSION
# Models: Linear Regression (spreads) + Logistic Regression (signal classifier)
#
# Improvements over previous version:
#   1. Chronological train/test split (no lookahead bias)
#   2. Expanding-window mean/std for labeling (no future data leakage)
#   3. Fixed convergence check (spread returns near mean, not just < std)
#   4. Rolling half-life (60-day window) — varies per spike, not constant per pair
#   5. Added 'days outside band' feature — consecutive days above threshold
#   6. Manual class weights based on actual class ratio (replaces 'balanced')
#   7. Tuned probability threshold (0.65) instead of default 0.5
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)


# =============================================================================
# CONFIG
# =============================================================================

pairs = [
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

start      = '2019-01-01'
end        = '2024-01-01'
MIN_WINDOW = 60    # ~3 months before expanding window kicks in
HL_WINDOW  = 60    # rolling window for half-life computation
THRESHOLD  = 0.65  # probability threshold — only fire on high-confidence signals


# =============================================================================
# 1. DATA DOWNLOAD
# =============================================================================

all_tickers = list(set(t for pair in pairs for t in pair))
raw_data    = yf.download(all_tickers, start=start, end=end)['Close']
raw_data.sort_index(inplace=True)

print(raw_data.head())
print(f"Shape: {raw_data.shape}")


# =============================================================================
# 2. PAIR CORRELATIONS
# =============================================================================

print("\nPair Correlations:")
print("-" * 35)
for s1, s2 in pairs:
    corr = raw_data[s1].corr(raw_data[s2])
    print(f"{s1:4s} / {s2:4s}: {corr:.4f}")


# =============================================================================
# 3. SPREADS via LINEAR REGRESSION
# =============================================================================

spreads = {}

for s1, s2 in pairs:
    X = raw_data[s1].values.reshape(-1, 1)
    y = raw_data[s2].values

    model  = LinearRegression()
    model.fit(X, y)
    spread = y - model.predict(X)

    spreads[f"{s1}/{s2}"] = pd.Series(spread, index=raw_data.index)

print("\nSpreads calculated for all pairs.")


# =============================================================================
# 4. ROLLING HALF-LIFE OF MEAN REVERSION
#
#    FIX vs previous: computed on a rolling 60-day window ending at day i,
#    so the half-life varies per spike rather than being one constant per pair.
#    This makes it a genuinely informative per-event feature.
#
#    Formula: fit delta_spread = alpha + beta * spread_lag on the window,
#             half-life = -ln(2) / beta
# =============================================================================

def rolling_halflife(spread_series, window=60):
    """
    Compute rolling half-life of mean reversion.
    Returns a Series aligned with spread_series index.
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

print("\nComputing rolling half-lives (takes ~30s)...")
rolling_hl = {}
for s1, s2 in pairs:
    pair_name             = f"{s1}/{s2}"
    rolling_hl[pair_name] = rolling_halflife(spreads[pair_name], window=HL_WINDOW)
    median_hl             = rolling_hl[pair_name].replace(999.0, np.nan).median()
    print(f"  {pair_name}: median rolling half-life = {median_hl:.1f} days")


# =============================================================================
# 5. LABELING — expanding window, fixed convergence check
# =============================================================================

labels = {}

for pair_name, spread in spreads.items():
    pair_labels = pd.Series(0, index=spread.index)

    for i in range(MIN_WINDOW, len(spread) - 20):
        history = spread.iloc[:i]
        mean    = history.mean()
        std     = history.std()

        if std == 0:
            continue

        upper = mean + 2 * std
        lower = mean - 2 * std
        val   = spread.iloc[i]

        if val > upper or val < lower:
            future    = spread.iloc[i+1 : i+21]
            converged = ((future - mean).abs() < 0.5 * std).any()
            pair_labels.iloc[i] = 1 if converged else 0

    labels[pair_name] = pair_labels

print("\nLabeling done (expanding window).")
for pair_name in labels:
    n_sig   = labels[pair_name].sum()
    n_noise = (labels[pair_name] == 0).sum()
    print(f"  {pair_name}: signals={n_sig}, noise={n_noise}")


# =============================================================================
# 6. FEATURE ENGINEERING
#
#    Features per spike event:
#      - zscore       : how many std devs the spread is from the mean
#      - velocity     : change in spread from previous day (momentum)
#      - rolling_hl   : half-life computed on 60-day window ending at day i
#      - days_outside : consecutive days the spread has been outside the band
#
#    'days_outside' is new — a spread outside for 10 days is much less likely
#    to converge than one that just crossed the threshold today.
# =============================================================================

all_features   = []
all_labels     = []
all_dates      = []
all_pairs_feat = []

for pair_name, spread in spreads.items():
    hl_series = rolling_hl[pair_name]

    for i in range(MIN_WINDOW + 1, len(spread) - 20):
        history = spread.iloc[:i]
        mean    = history.mean()
        std     = history.std()

        if std == 0:
            continue

        val   = spread.iloc[i]
        upper = mean + 2 * std
        lower = mean - 2 * std

        if val > upper or val < lower:
            velocity = val - spread.iloc[i-1]
            zscore   = (val - mean) / std
            hl       = hl_series.iloc[i]

            # Count consecutive days outside the band (look back up to 20 days)
            days_outside = 0
            for j in range(i - 1, max(i - 21, MIN_WINDOW - 1), -1):
                pv = spread.iloc[j]
                ph = spread.iloc[:j]
                if len(ph) < 2:
                    break
                pm = ph.mean()
                ps = ph.std()
                if ps == 0:
                    break
                if pv > pm + 2*ps or pv < pm - 2*ps:
                    days_outside += 1
                else:
                    break

            all_features.append([zscore, velocity, hl, days_outside])
            all_labels.append(labels[pair_name].iloc[i])
            all_dates.append(spread.index[i])
            all_pairs_feat.append(pair_name)

X_all = pd.DataFrame(all_features,
                     columns=['zscore', 'velocity', 'rolling_hl', 'days_outside'],
                     index=all_dates)
y_all = pd.Series(all_labels, index=all_dates)

print(f"\nTotal spike events: {len(y_all)}")
print(f"Real signals: {y_all.sum()}  |  Noise: {(y_all == 0).sum()}")


# =============================================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT + LOGISTIC REGRESSION
# =============================================================================

split_idx  = int(len(X_all) * 0.8)
X_train    = X_all.iloc[:split_idx]
X_test     = X_all.iloc[split_idx:]
y_train    = y_all.iloc[:split_idx]
y_test     = y_all.iloc[split_idx:]

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Manual class weight — sqrt of ratio softens overcompensation vs 'balanced'
n_noise   = (y_train == 0).sum()
n_signal  = (y_train == 1).sum()
ratio     = n_noise / n_signal if n_signal > 0 else 1.0
weight    = float(np.sqrt(ratio))
class_weights = {0: 1.0, 1: weight}

print(f"\nClass weight for signal: {weight:.2f}  (noise:signal ratio = {ratio:.1f})")

log_model = LogisticRegression(class_weight=class_weights, max_iter=1000)
log_model.fit(X_train_sc, y_train)

# Tuned threshold — only predict signal when model is highly confident
probs_test = log_model.predict_proba(X_test_sc)[:, 1]
y_pred     = (probs_test > THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Accuracy (out-of-sample, threshold={THRESHOLD}): {accuracy:.4f}")
print(classification_report(y_test, y_pred, zero_division=0))


# =============================================================================
# GRAPH SET 1 — LINEAR REGRESSION: PRICE SCATTER + FITTED LINE (all 12 pairs)
# =============================================================================

fig, axes = plt.subplots(4, 3, figsize=(18, 22))
axes = axes.flatten()

for idx, (s1, s2) in enumerate(pairs):
    ax     = axes[idx]
    X_plot = raw_data[s1].values.reshape(-1, 1)
    y_plot = raw_data[s2].values

    lr     = LinearRegression().fit(X_plot, y_plot)
    x_line = np.linspace(X_plot.min(), X_plot.max(), 300).reshape(-1, 1)
    y_line = lr.predict(x_line)
    r2     = lr.score(X_plot, y_plot)
    beta   = lr.coef_[0]

    ax.scatter(X_plot, y_plot, alpha=0.2, s=8, color='steelblue', label='Daily prices')
    ax.plot(x_line, y_line, color='red', linewidth=2,
            label=f'β={beta:.3f}, R²={r2:.3f}')
    ax.set_title(f'{s1} vs {s2}', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{s1} Price ($)', fontsize=9)
    ax.set_ylabel(f'{s2} Price ($)', fontsize=9)
    ax.legend(fontsize=8)

fig.suptitle('Graph Set 1 — Linear Regression: All 12 Pairs (2019–2024)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# =============================================================================
# GRAPH SET 2 — LOGISTIC REGRESSION DECISION BOUNDARIES (all 12 pairs)
#   Per-pair logistic regression on that pair's spike events.
#   Axes: zscore vs velocity. rolling_hl and days_outside held at 0 (scaled mean).
#   Tuned threshold applied to both background shading and dot classification.
# =============================================================================

fig, axes = plt.subplots(4, 3, figsize=(18, 22))
axes = axes.flatten()

for idx, (s1, s2) in enumerate(pairs):
    ax        = axes[idx]
    pair_name = f"{s1}/{s2}"
    spread    = spreads[pair_name]
    hl_series = rolling_hl[pair_name]

    pair_features    = []
    pair_labels_list = []

    for i in range(MIN_WINDOW + 1, len(spread) - 20):
        history = spread.iloc[:i]
        mean    = history.mean()
        std     = history.std()
        if std == 0:
            continue

        val   = spread.iloc[i]
        upper = mean + 2 * std
        lower = mean - 2 * std

        if val > upper or val < lower:
            velocity = val - spread.iloc[i-1]
            zscore   = (val - mean) / std
            hl       = hl_series.iloc[i]

            days_outside = 0
            for j in range(i - 1, max(i - 21, MIN_WINDOW - 1), -1):
                pv = spread.iloc[j]
                ph = spread.iloc[:j]
                if len(ph) < 2:
                    break
                pm = ph.mean()
                ps = ph.std()
                if ps == 0:
                    break
                if pv > pm + 2*ps or pv < pm - 2*ps:
                    days_outside += 1
                else:
                    break

            pair_features.append([zscore, velocity, hl, days_outside])
            pair_labels_list.append(labels[pair_name].iloc[i])

    if len(pair_features) < 10:
        ax.set_title(f'{pair_name} (insufficient data)')
        continue

    X_pair = np.array(pair_features)
    y_pair = np.array(pair_labels_list)

    if len(np.unique(y_pair)) < 2:
        ax.set_title(f'{pair_name} (only one class — skipped)')
        continue

    n0 = (y_pair == 0).sum()
    n1 = (y_pair == 1).sum()
    r  = n0 / n1 if n1 > 0 else 1.0
    cw = {0: 1.0, 1: float(np.sqrt(r))}

    p_scaler  = StandardScaler()
    X_pair_sc = p_scaler.fit_transform(X_pair)

    p_log = LogisticRegression(class_weight=cw, max_iter=1000)
    p_log.fit(X_pair_sc, y_pair)

    z_col = X_pair_sc[:, 0]
    v_col = X_pair_sc[:, 1]

    z_min, z_max = z_col.min() - 0.5, z_col.max() + 0.5
    v_min, v_max = v_col.min() - 0.5, v_col.max() + 0.5

    zz, vv = np.meshgrid(np.linspace(z_min, z_max, 300),
                         np.linspace(v_min, v_max, 300))

    grid_input  = np.column_stack([zz.ravel(), vv.ravel(),
                                   np.zeros(zz.ravel().shape),
                                   np.zeros(zz.ravel().shape)])
    grid_probs  = p_log.predict_proba(grid_input)[:, 1]
    Z           = (grid_probs > THRESHOLD).astype(int).reshape(zz.shape)

    ax.contourf(zz, vv, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5],
                colors=['#ff9999', '#99ff99'])
    ax.contour(zz, vv, Z, levels=[0.5], colors='black', linewidths=1.5)

    pair_probs = p_log.predict_proba(X_pair_sc)[:, 1]
    pair_pred  = (pair_probs > THRESHOLD).astype(int)

    sig   = y_pair == 1
    noise = y_pair == 0
    ax.scatter(z_col[sig],   v_col[sig],   color='green', s=25,
               alpha=0.7, zorder=5, label='Signal')
    ax.scatter(z_col[noise], v_col[noise], color='red',   s=25,
               alpha=0.7, zorder=5, label='Noise')

    pair_acc = accuracy_score(y_pair, pair_pred)
    ax.set_title(f'{pair_name}  (acc={pair_acc:.2f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Z-score (scaled)', fontsize=8)
    ax.set_ylabel('Velocity (scaled)', fontsize=8)
    ax.legend(fontsize=7)

fig.suptitle('Graph Set 2 — Logistic Regression Decision Boundaries (All 12 Pairs)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# =============================================================================
# GRAPH SET 3 — SPREAD OVER TIME with threshold bands + signal dots (all 12)
# =============================================================================

fig, axes = plt.subplots(4, 3, figsize=(20, 24))
axes = axes.flatten()

for idx, (s1, s2) in enumerate(pairs):
    ax        = axes[idx]
    pair_name = f"{s1}/{s2}"
    spread    = spreads[pair_name]

    mean_vis = spread.mean()
    std_vis  = spread.std()
    upper    = mean_vis + 2 * std_vis
    lower    = mean_vis - 2 * std_vis

    ax.plot(spread.index, spread.values, color='steelblue',
            linewidth=0.8, alpha=0.9, label='Spread')
    ax.axhline(mean_vis, color='black',  linewidth=1,   linestyle='--', label='Mean')
    ax.axhline(upper,    color='orange', linewidth=1,   linestyle='--', label='+2σ')
    ax.axhline(lower,    color='orange', linewidth=1,   linestyle='--', label='-2σ')
    ax.fill_between(spread.index, lower, upper, alpha=0.06, color='orange')

    pair_label_series = labels[pair_name]
    spike_mask        = (spread > upper) | (spread < lower)

    sig_idx   = spread.index[spike_mask & (pair_label_series == 1)]
    noise_idx = spread.index[spike_mask & (pair_label_series == 0)]

    ax.scatter(sig_idx,   spread[sig_idx],   color='green', s=20,
               zorder=5, label='Signal', alpha=0.8)
    ax.scatter(noise_idx, spread[noise_idx], color='red',   s=20,
               zorder=5, label='Noise',  alpha=0.8)

    ax.set_title(f'{pair_name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=8)
    ax.set_ylabel('Spread ($)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.legend(fontsize=7)

fig.suptitle('Graph Set 3 — Spread Over Time with Signal Labels (All 12 Pairs)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()


# =============================================================================
# GRAPH SET 4 — GLOBAL MODEL PERFORMANCE (out-of-sample)
#   (a) Confusion matrix
#   (b) Feature importance (all 4 features)
#   (c) Precision / Recall by class
#   (d) Probability distribution with both thresholds marked
# =============================================================================

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- (a) Confusion Matrix ---
ax_cm = fig.add_subplot(gs[0, 0])
cm    = confusion_matrix(y_test, y_pred)
disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Noise', 'Signal'])
disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
ax_cm.set_title('(a) Confusion Matrix — Out-of-Sample Test Set',
                fontsize=11, fontweight='bold')

# --- (b) Feature Importance ---
ax_fi      = fig.add_subplot(gs[0, 1])
feat_names = ['Z-score', 'Velocity', 'Rolling HL', 'Days Outside']
coefs      = log_model.coef_[0]
colors_fi  = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefs]

bars = ax_fi.barh(feat_names, np.abs(coefs), color=colors_fi,
                  edgecolor='black', linewidth=0.6)
ax_fi.set_xlabel('|Coefficient| (scaled)', fontsize=9)
ax_fi.set_title('(b) Feature Importance (Logistic Regression Coefficients)',
                fontsize=11, fontweight='bold')

for bar, c in zip(bars, coefs):
    sign = '+' if c > 0 else '-'
    ax_fi.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
               f'{sign}{abs(c):.3f}', va='center', fontsize=9)

ax_fi.set_xlim(0, max(np.abs(coefs)) * 1.4)
legend_els = [Patch(facecolor='#2ecc71', label='Positive (-> signal)'),
              Patch(facecolor='#e74c3c', label='Negative (-> noise)')]
ax_fi.legend(handles=legend_els, fontsize=8)

# --- (c) Precision / Recall ---
ax_pr  = fig.add_subplot(gs[1, 0])
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

classes = ['Noise (0)', 'Signal (1)']
prec    = [report['0']['precision'], report['1']['precision']]
rec     = [report['0']['recall'],    report['1']['recall']]

x     = np.arange(len(classes))
width = 0.35

ax_pr.bar(x - width/2, prec, width, label='Precision', color='#3498db', edgecolor='black')
ax_pr.bar(x + width/2, rec,  width, label='Recall',    color='#e67e22', edgecolor='black')
ax_pr.set_xticks(x)
ax_pr.set_xticklabels(classes, fontsize=10)
ax_pr.set_ylim(0, 1.15)
ax_pr.set_ylabel('Score', fontsize=9)
ax_pr.set_title('(c) Precision & Recall by Class', fontsize=11, fontweight='bold')
ax_pr.legend(fontsize=9)
ax_pr.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

for i, (p, r) in enumerate(zip(prec, rec)):
    ax_pr.text(i - width/2, p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
    ax_pr.text(i + width/2, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)

# --- (d) Probability Distribution ---
ax_pd = fig.add_subplot(gs[1, 1])

sig_probs   = probs_test[y_test == 1]
noise_probs = probs_test[y_test == 0]

ax_pd.hist(noise_probs, bins=25, alpha=0.6, color='#e74c3c',
           label='True Noise',  edgecolor='black', linewidth=0.4)
ax_pd.hist(sig_probs,   bins=25, alpha=0.6, color='#2ecc71',
           label='True Signal', edgecolor='black', linewidth=0.4)
ax_pd.axvline(0.5,       color='gray',  linestyle='--', linewidth=1.0,
              label='Default threshold (0.5)')
ax_pd.axvline(THRESHOLD, color='black', linestyle='--', linewidth=1.5,
              label=f'Tuned threshold ({THRESHOLD})')
ax_pd.set_xlabel('P(Signal)', fontsize=9)
ax_pd.set_ylabel('Count', fontsize=9)
ax_pd.set_title('(d) Predicted Probability Distribution', fontsize=11, fontweight='bold')
ax_pd.legend(fontsize=8)

fig.suptitle('Graph Set 4 — Global Model Performance (Out-of-Sample)',
             fontsize=15, fontweight='bold')
plt.show()

print("\nDone. All 4 graph sets displayed.")