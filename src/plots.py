"""
Plotting module.

Four graph sets:
1. Linear regression scatter for each pair
2. Logistic regression decision boundaries per pair
3. Spreads over time with signal/noise labels
4. Global model performance (confusion matrix, feature importance, P/R, prob dist)

All plots saved to results/ as PNGs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)


RESULTS_DIR = 'results'


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_regression_scatter(data, pairs, save=True):
    """Graph Set 1: price scatter + fitted line for each pair."""
    _ensure_results_dir()
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    axes = axes.flatten()

    for idx, (s1, s2) in enumerate(pairs):
        ax     = axes[idx]
        X_plot = data[s1].values.reshape(-1, 1)
        y_plot = data[s2].values

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

    fig.suptitle('Graph Set 1 — Linear Regression: All Pairs',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, '01_regression_scatter.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"  saved: {path}")
    plt.close(fig)


def plot_decision_boundaries(spreads, rolling_hl, labels, pairs,
                              min_window=60, threshold=0.65, save=True):
    """Graph Set 2: per-pair logistic regression decision boundaries."""
    _ensure_results_dir()
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    axes = axes.flatten()

    for idx, (s1, s2) in enumerate(pairs):
        ax        = axes[idx]
        pair_name = f"{s1}/{s2}"
        spread    = spreads[pair_name]
        hl_series = rolling_hl[pair_name]

        pair_features    = []
        pair_labels_list = []

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

        grid_input = np.column_stack([zz.ravel(), vv.ravel(),
                                      np.zeros(zz.ravel().shape),
                                      np.zeros(zz.ravel().shape)])
        grid_probs = p_log.predict_proba(grid_input)[:, 1]
        Z = (grid_probs > threshold).astype(int).reshape(zz.shape)

        ax.contourf(zz, vv, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5],
                    colors=['#ff9999', '#99ff99'])
        ax.contour(zz, vv, Z, levels=[0.5], colors='black', linewidths=1.5)

        pair_probs = p_log.predict_proba(X_pair_sc)[:, 1]
        pair_pred  = (pair_probs > threshold).astype(int)

        sig   = y_pair == 1
        noise = y_pair == 0
        ax.scatter(z_col[sig],   v_col[sig],   color='green', s=25,
                   alpha=0.7, zorder=5, label='Signal')
        ax.scatter(z_col[noise], v_col[noise], color='red', s=25,
                   alpha=0.7, zorder=5, label='Noise')

        pair_acc = accuracy_score(y_pair, pair_pred)
        ax.set_title(f'{pair_name}  (acc={pair_acc:.2f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Z-score (scaled)', fontsize=8)
        ax.set_ylabel('Velocity (scaled)', fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle('Graph Set 2 — Logistic Regression Decision Boundaries',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, '02_decision_boundaries.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"  saved: {path}")
    plt.close(fig)


def plot_spreads_with_signals(spreads, labels, pairs, save=True):
    """Graph Set 3: spreads over time with signal/noise labels."""
    _ensure_results_dir()
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
        ax.axhline(mean_vis, color='black',  linewidth=1, linestyle='--', label='Mean')
        ax.axhline(upper,    color='orange', linewidth=1, linestyle='--', label='+2σ')
        ax.axhline(lower,    color='orange', linewidth=1, linestyle='--', label='-2σ')
        ax.fill_between(spread.index, lower, upper, alpha=0.06, color='orange')

        pair_label_series = labels[pair_name]
        spike_mask        = (spread > upper) | (spread < lower)

        sig_idx   = spread.index[spike_mask & (pair_label_series == 1)]
        noise_idx = spread.index[spike_mask & (pair_label_series == 0)]

        ax.scatter(sig_idx,   spread[sig_idx],   color='green', s=20,
                   zorder=5, label='Signal', alpha=0.8)
        ax.scatter(noise_idx, spread[noise_idx], color='red', s=20,
                   zorder=5, label='Noise', alpha=0.8)

        ax.set_title(f'{pair_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel('Spread ($)', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle('Graph Set 3 — Spread Over Time with Signal Labels',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save:
        path = os.path.join(RESULTS_DIR, '03_spreads_with_signals.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"  saved: {path}")
    plt.close(fig)


def plot_model_performance(train_result, save=True):
    """Graph Set 4: confusion matrix, feature importance, P/R, prob distribution."""
    _ensure_results_dir()

    model      = train_result['model']
    y_test     = train_result['y_test']
    y_pred     = train_result['y_pred']
    probs_test = train_result['probs_test']
    threshold  = train_result['threshold']

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # (a) Confusion Matrix
    ax_cm = fig.add_subplot(gs[0, 0])
    cm    = confusion_matrix(y_test, y_pred)
    disp  = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Noise', 'Signal'])
    disp.plot(ax=ax_cm, colorbar=False, cmap='Blues')
    ax_cm.set_title('(a) Confusion Matrix — Out-of-Sample',
                    fontsize=11, fontweight='bold')

    # (b) Feature Importance
    ax_fi      = fig.add_subplot(gs[0, 1])
    feat_names = ['Z-score', 'Velocity', 'Rolling HL', 'Days Outside']
    coefs      = model.coef_[0]
    colors_fi  = ['#2ecc71' if c > 0 else '#e74c3c' for c in coefs]

    bars = ax_fi.barh(feat_names, np.abs(coefs), color=colors_fi,
                      edgecolor='black', linewidth=0.6)
    ax_fi.set_xlabel('|Coefficient| (scaled)', fontsize=9)
    ax_fi.set_title('(b) Feature Importance', fontsize=11, fontweight='bold')

    for bar, c in zip(bars, coefs):
        sign = '+' if c > 0 else '-'
        ax_fi.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                   f'{sign}{abs(c):.3f}', va='center', fontsize=9)

    ax_fi.set_xlim(0, max(np.abs(coefs)) * 1.4)
    legend_els = [Patch(facecolor='#2ecc71', label='Positive (→ signal)'),
                  Patch(facecolor='#e74c3c', label='Negative (→ noise)')]
    ax_fi.legend(handles=legend_els, fontsize=8)

    # (c) Precision / Recall
    ax_pr  = fig.add_subplot(gs[1, 0])
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    classes = ['Noise (0)', 'Signal (1)']
    prec    = [report['0']['precision'], report['1']['precision']]
    rec     = [report['0']['recall'],    report['1']['recall']]

    x     = np.arange(len(classes))
    width = 0.35

    ax_pr.bar(x - width / 2, prec, width, label='Precision',
              color='#3498db', edgecolor='black')
    ax_pr.bar(x + width / 2, rec,  width, label='Recall',
              color='#e67e22', edgecolor='black')
    ax_pr.set_xticks(x)
    ax_pr.set_xticklabels(classes, fontsize=10)
    ax_pr.set_ylim(0, 1.15)
    ax_pr.set_ylabel('Score', fontsize=9)
    ax_pr.set_title('(c) Precision & Recall by Class',
                    fontsize=11, fontweight='bold')
    ax_pr.legend(fontsize=9)
    ax_pr.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

    for i, (p, r) in enumerate(zip(prec, rec)):
        ax_pr.text(i - width / 2, p + 0.02, f'{p:.2f}', ha='center', fontsize=9)
        ax_pr.text(i + width / 2, r + 0.02, f'{r:.2f}', ha='center', fontsize=9)

    # (d) Probability Distribution
    ax_pd = fig.add_subplot(gs[1, 1])

    sig_probs   = probs_test[y_test == 1]
    noise_probs = probs_test[y_test == 0]

    ax_pd.hist(noise_probs, bins=25, alpha=0.6, color='#e74c3c',
               label='True Noise',  edgecolor='black', linewidth=0.4)
    ax_pd.hist(sig_probs,   bins=25, alpha=0.6, color='#2ecc71',
               label='True Signal', edgecolor='black', linewidth=0.4)
    ax_pd.axvline(0.5,       color='gray',  linestyle='--', linewidth=1.0,
                  label='Default threshold (0.5)')
    ax_pd.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                  label=f'Tuned threshold ({threshold})')
    ax_pd.set_xlabel('P(Signal)', fontsize=9)
    ax_pd.set_ylabel('Count', fontsize=9)
    ax_pd.set_title('(d) Predicted Probability Distribution',
                    fontsize=11, fontweight='bold')
    ax_pd.legend(fontsize=8)

    fig.suptitle('Graph Set 4 — Global Model Performance (Out-of-Sample)',
                 fontsize=15, fontweight='bold')

    if save:
        path = os.path.join(RESULTS_DIR, '04_model_performance.png')
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"  saved: {path}")
    plt.close(fig)