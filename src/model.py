"""
Model training module.

Performs:
- Chronological train/test split (no lookahead bias)
- Standard scaling
- Logistic regression with manual class weights
- Tuned probability threshold for prediction
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_classifier(X, y, test_size=0.2, threshold=0.65, verbose=True):
    """
    Train logistic regression on a feature matrix with chronological split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows ordered by date).
    y : pd.Series
        Binary labels aligned with X.
    test_size : float
        Fraction of data reserved for out-of-sample testing.
    threshold : float
        Probability threshold for predicting class 1 (signal).
    verbose : bool
        Print summary stats if True.

    Returns
    -------
    dict
        {
            'model': fitted LogisticRegression,
            'scaler': fitted StandardScaler,
            'X_train', 'X_test', 'y_train', 'y_test',
            'y_pred', 'probs_test',
            'accuracy', 'threshold',
            'class_weights',
        }
    """
    # Chronological split
    split_idx = int(len(X) * (1 - test_size))
    X_train   = X.iloc[:split_idx]
    X_test    = X.iloc[split_idx:]
    y_train   = y.iloc[:split_idx]
    y_test    = y.iloc[split_idx:]

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Manual class weight: sqrt of ratio softens overcompensation vs 'balanced'
    n_noise  = (y_train == 0).sum()
    n_signal = (y_train == 1).sum()
    ratio    = n_noise / n_signal if n_signal > 0 else 1.0
    weight   = float(np.sqrt(ratio))
    class_weights = {0: 1.0, 1: weight}

    # Fit
    model = LogisticRegression(class_weight=class_weights, max_iter=1000)
    model.fit(X_train_sc, y_train)

    # Predict with tuned threshold
    probs_test = model.predict_proba(X_test_sc)[:, 1]
    y_pred     = (probs_test > threshold).astype(int)
    accuracy   = accuracy_score(y_test, y_pred)

    if verbose:
        print(f"\nClass weight for signal: {weight:.2f}  "
              f"(noise:signal ratio = {ratio:.1f})")
        print(f"Out-of-sample accuracy (threshold={threshold}): {accuracy:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'probs_test': probs_test,
        'accuracy': accuracy,
        'threshold': threshold,
        'class_weights': class_weights,
    }