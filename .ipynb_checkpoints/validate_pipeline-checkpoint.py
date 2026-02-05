# validate_pipeline.py
"""
Validation utilities for the volatility forecasting pipeline (updated).

This module provides essential leakage detectors and evaluation helpers, aligned to the
current rolling-ML pipeline variable names.

Main functions:
 - sanitize_features
 - check_ids_not_in_features
 - check_time_splits
 - day_level_leak_scan
 - check_merge_and_duplicates
 - check_rolling_lagged
 - compute_spearman_stats
 - permutation_importance_feature
 - paired_bootstrap_spearman
 - run_full_validation

Usage (after rolling ML block):

    from validate_pipeline import run_full_validation

    diagnostics = run_full_validation(
        df=df,
        features=features,
        pred_df=pred_df,
        rolling_train_spearman=rolling_train_spearman,
        rolling_test_spearman=rolling_test_spearman,
        unique_days=unique_days,
        train_window=train_window,
        test_horizon=test_horizon,
        model=model,            # optional
        perm_features=['GAS_RET_60m'],
        out_csv='outputs/validation_summary.csv'
    )

The function returns a dictionary `diagnostics` with the main checks and metrics.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging
from typing import List, Tuple, Dict, Any, Optional

# xgboost import is optional (only required for permutation checks when model provided)
try:
    import xgboost as xgb
except Exception:
    xgb = None

log = logging.getLogger("validate_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -------------------------
# Helper: sanitize features (lightweight)
# -------------------------

def sanitize_features(X: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce DataFrame X to numeric features safe for an XGBoost DMatrix.
    Returns (X_numeric, dropped_columns). Minimal: converts datetimes, coerces to numeric,
    drops fully-nonnumeric columns, fills NaNs with 0 and casts to float32.
    """
    X = X.copy()
    dropped: List[str] = []

    for c in X.columns:
        if np.issubdtype(X[c].dtype, np.datetime64):
            if verbose:
                log.info(f"Converting datetime column to int: {c}")
            X[c] = X[c].astype('int64') // 10**9

    Xn = X.apply(pd.to_numeric, errors='coerce')
    bad_cols = [c for c in Xn.columns if Xn[c].isna().all()]
    if bad_cols:
        if verbose:
            log.warning(f"Dropping non-numeric columns: {bad_cols}")
        Xn = Xn.drop(columns=bad_cols)
        dropped.extend(bad_cols)

    Xn = Xn.fillna(0).astype(np.float32)
    return Xn, dropped


# -------------------------
# Basic checks
# -------------------------

def check_ids_not_in_features(features: List[str], id_cols: Optional[List[str]] = None) -> List[str]:
    """Return list of id-like columns present in features (should be empty).
    Typical id_cols: ['ID','DAY_ID','COUNTRY','index']
    """
    if id_cols is None:
        id_cols = ['ID', 'DAY_ID', 'COUNTRY', 'index']
    bad = [c for c in features if c in id_cols]
    if bad:
        log.warning(f"Identifier columns present in features: {bad}")
    else:
        log.info("No identifier columns present in features")
    return bad


def check_time_splits(train_days: List[Any], holdout_days: List[Any]) -> Dict[str,int]:
    """Check that train_days and holdout_days are disjoint. Return overlap counts."""
    i = len(set(train_days).intersection(holdout_days))
    if i > 0:
        log.warning(f"Train/holdout overlap detected: {i} days")
    else:
        log.info("Train and holdout days are disjoint")
    return {'train_holdout_overlap': i}


def day_level_leak_scan(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Return features that are constant across countries within a day (suspicious).
    For each feature compute median unique values per DAY_ID; if <= 1 flag it.
    """
    leaky = []
    if 'DAY_ID' not in df.columns:
        log.warning('DAY_ID not in df — cannot perform day-level leak scan')
        return leaky
    for f in features:
        if f not in df.columns:
            continue
        try:
            uniq = df.groupby('DAY_ID')[f].nunique().median()
            if uniq <= 1:
                leaky.append(f)
        except Exception:
            continue
    if leaky:
        log.warning(f"Per-day constant features (examine): {leaky}")
    else:
        log.info("No per-day constant features detected")
    return leaky


def check_merge_and_duplicates(pred_df: pd.DataFrame) -> Dict[str,int]:
    """Check duplicates in prediction DataFrame (should be none on DAY_ID+COUNTRY).
    Returns dict with counts.
    """
    if pred_df is None:
        raise ValueError('pred_df is None')
    dup = int(pred_df.duplicated(subset=['DAY_ID','COUNTRY']).sum())
    nrows = int(pred_df.shape[0])
    log.info(f"pred_df rows={nrows}, duplicates on (DAY_ID,COUNTRY)={dup}")
    return {'pred_rows': nrows, 'duplicates': dup}


# -------------------------
# Rolling-lag sanity check
# -------------------------

def check_rolling_lagged(df: pd.DataFrame, raw_col: str, engineered_col: str, window: int = 60) -> float:
    """Simple sanity check: compute rolling(window).mean().shift(1) from raw_col and compare to engineered_col.
    Returns Pearson correlation (close to 1.0 expected if engineered_col was computed correctly using past data).
    """
    if raw_col not in df.columns or engineered_col not in df.columns:
        log.warning(f"Columns missing for lag check: {raw_col} or {engineered_col}")
        return np.nan
    check = df.groupby('COUNTRY')[raw_col].transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
    sub = pd.concat([check, df[engineered_col]], axis=1).dropna()
    if sub.shape[0] == 0:
        return np.nan
    corr = sub.corr().iloc[0,1]
    log.info(f"Lag-check corr for {engineered_col}: {corr:.4f}")
    return float(corr)


def _resolve_col(df, base):
    if base in df.columns:
        return base
    for c in df.columns:
        if c.startswith(base + '_'):
            return c
    raise KeyError(f'{base} not found in columns')

# -------------------------
# Spearman / per-day stats
# -------------------------

def compute_spearman_stats(df, pred_col='pred_residual', true_col='true_residual'):
    pred_col = _resolve_col(df, pred_col)
    true_col = _resolve_col(df, true_col)

    mask = df[pred_col].notna()
    if mask.sum() == 0:
        raise ValueError('No predictions found in df')
    pooled = float(spearmanr(df.loc[mask, pred_col], df.loc[mask, true_col]).correlation)

    day_rhos = []
    for d in df.loc[mask, 'DAY_ID'].unique():
        sub = df[(df['DAY_ID'] == d) & mask]
        if sub.shape[0] <= 1:
            continue
        r = spearmanr(sub[pred_col], sub[true_col]).correlation
        if not np.isnan(r):
            day_rhos.append(float(r))
    day_rhos = np.array(day_rhos) if len(day_rhos) else np.array([])

    res = {
        'pooled_spearman': pooled,
        'per_day_mean': float(np.nanmean(day_rhos)) if day_rhos.size else np.nan,
        'per_day_median': float(np.nanmedian(day_rhos)) if day_rhos.size else np.nan,
        'per_day_count': int(day_rhos.size),
        'per_day_array': day_rhos
    }
    log.info(f"Pooled spearman: {pooled:.4f}, per-day mean: {res['per_day_mean']:.4f} (n={res['per_day_count']})")
    return res


# -------------------------
# Permutation importance (single feature)
# -------------------------

def permutation_importance_feature(model: Any, df_hold: pd.DataFrame, features: List[str], feature: str,
                                   nperm: int = 200) -> Dict[str, Any]:
    """Permute a single feature on holdout and report mean rho and drop vs base.
    - model: xgboost or any model with predict(DMatrix) method
    - df_hold: DataFrame of holdout rows (contains features and true_residual)
    - features: list of feature names (in correct order)
    Returns dict with base_rho, perm_mean, perm_std, drop
    """
    if xgb is None:
        raise RuntimeError('xgboost not available for permutation checks')
    if feature not in features:
        raise ValueError(f"Feature {feature} not in features")

    X_hold = df_hold[features].copy()
    y_hold = df_hold['true_residual'].values
    X_hold_s, _ = sanitize_features(X_hold, verbose=False)
    # ensure order
    X_hold_s = X_hold_s[[c for c in features if c in X_hold_s.columns]]

    d = xgb.DMatrix(X_hold_s)
    base_pred = model.predict(d)
    base_rho = float(spearmanr(base_pred, y_hold).correlation)

    perm_rhos = []
    Xp = X_hold_s.copy()
    for _ in range(nperm):
        Xp[feature] = np.random.permutation(X_hold_s[feature].values)
        rp = model.predict(xgb.DMatrix(Xp))
        perm_rhos.append(float(spearmanr(rp, y_hold).correlation))
    perm_rhos = np.array(perm_rhos)
    res = {
        'feature': feature,
        'base_rho': base_rho,
        'perm_mean': float(np.nanmean(perm_rhos)),
        'perm_std': float(np.nanstd(perm_rhos)),
        'drop': float(base_rho - np.nanmean(perm_rhos))
    }
    log.info(f"Permute {feature}: base_rho={res['base_rho']:.4f}, perm_mean={res['perm_mean']:.4f}, drop={res['drop']:.4f}")
    return res


# -------------------------
# Paired bootstrap for model comparison
# -------------------------

def paired_bootstrap_spearman(preds_a: np.ndarray, preds_b: np.ndarray, y: np.ndarray, day_ids: np.ndarray,
                              n_iter: int = 1000) -> Tuple[np.ndarray, float]:
    """Return 95% CI and mean of (rho_a - rho_b) under paired bootstrap over days.
    preds_a, preds_b, y, day_ids must be aligned arrays of same length.
    """
    unique_days = np.unique(day_ids)
    diffs = []
    for _ in range(n_iter):
        samp_days = np.random.choice(unique_days, size=len(unique_days), replace=True)
        mask = np.isin(day_ids, samp_days)
        try:
            ra = spearmanr(preds_a[mask], y[mask]).correlation
        except Exception:
            ra = np.nan
        try:
            rb = spearmanr(preds_b[mask], y[mask]).correlation
        except Exception:
            rb = np.nan
        diffs.append(ra - rb)
    diffs = np.array([d for d in diffs if not np.isnan(d)])
    if diffs.size == 0:
        return np.array([np.nan, np.nan, np.nan]), np.nan
    ci = np.percentile(diffs, [2.5, 50, 97.5])
    return ci, float(np.mean(diffs))


# -------------------------
# High-level runner (updated)
# -------------------------

def run_full_validation(
    df: pd.DataFrame,
    features: List[str],
    pred_df: pd.DataFrame,
    rolling_train_spearman: Optional[List[float]] = None,
    rolling_test_spearman: Optional[List[float]] = None,
    unique_days: Optional[np.ndarray] = None,
    train_window: Optional[int] = None,
    test_horizon: Optional[int] = None,
    model: Optional[Any] = None,
    perm_features: Optional[List[str]] = None,
    out_csv: Optional[str] = None
) -> Dict[str, Any]:
    """Run the recommended validation sequence using current rolling-ML variables.

    Required inputs:
      - df: main dataframe with DAY_ID, COUNTRY and true_residual
      - features: list of features used by the model
      - pred_df: DataFrame produced by rolling predictions (DAY_ID, COUNTRY, pred_residual, true_residual)

    Optional:
      - rolling_train_spearman, rolling_test_spearman: lists produced during rolling CV
      - unique_days, train_window, test_horizon: used to construct train/holdout splits for quick checks
      - model: trained xgboost model (optional) for permutation checks
      - perm_features: list of feature names to run permutation importance on
    """
    res: Dict[str, Any] = {}

    # 1) ID checks
    res['id_in_features'] = check_ids_not_in_features(features)

    # 2) split integrity if unique_days provided
    if unique_days is not None and test_horizon is not None:
        holdout_days = unique_days[-test_horizon:]
        train_days = unique_days[:-test_horizon]
        res['split_overlaps'] = check_time_splits(list(train_days), list(holdout_days))
    else:
        holdout_days = None

    # 3) pred_df sanity
    res['pred_df_checks'] = check_merge_and_duplicates(pred_df)

    # 4) day-level leak scan
    res['per_day_constant'] = day_level_leak_scan(df, features)

    # 5) compute spearman stats using merged data
    merged = df.copy()
    res['spearman_stats'] = compute_spearman_stats(merged, pred_col='pred_residual', true_col='true_residual')

    # 6) include rolling fold stats if available
    if rolling_train_spearman is not None:
        res['rolling_train_mean'] = float(np.nanmean(rolling_train_spearman))
    if rolling_test_spearman is not None:
        res['rolling_test_mean'] = float(np.nanmean(rolling_test_spearman))
        res['rolling_test_median'] = float(np.nanmedian(rolling_test_spearman))
        res['rolling_test_std'] = float(np.nanstd(rolling_test_spearman))

#    # 7) lag checks for common engineered features
#    lag_checks = {}
#    for eng in ['GAS_RET_60m', 'GAS_RET_30m', 'GAS_RET_60']:
#        if eng in df.columns and 'GAS_RET' in df.columns:
#            lag_checks[eng] = check_rolling_lagged(df, 'GAS_RET', eng, window=60)
#    res['lag_checks'] = lag_checks

    # 8) permutation importance (use features directly, no common_cols)
    if model is not None and perm_features is not None and len(perm_features) > 0:
        if xgb is None:
            log.warning('xgboost not available; skipping permutation checks')
        else:
            perm_res = {}
            # build holdout slice
            if holdout_days is not None:
                hold_mask = df['DAY_ID'].isin(holdout_days)
            else:
                hold_mask = merged['pred_residual'].notna()
            df_hold = merged.loc[hold_mask, ['DAY_ID','COUNTRY','pred_residual','true_residual'] + features].copy()
            for pf in perm_features:
                if pf in features:
                    try:
                        perm_res[pf] = permutation_importance_feature(model, df_hold, features, pf, nperm=200)
                    except Exception as e:
                        log.exception(f'Permutation check failed for {pf}: {e}')
            res['permutation'] = perm_res

    # 9) save summary CSV if requested
    if out_csv is not None:
        flat = {
            'id_in_features': str(res.get('id_in_features')),
            'pred_df_rows': res.get('pred_df_checks', {}).get('pred_rows'),
            'pred_df_duplicates': res.get('pred_df_checks', {}).get('duplicates'),
            'pooled_spearman': res.get('spearman_stats', {}).get('pooled_spearman'),
            'per_day_mean': res.get('spearman_stats', {}).get('per_day_mean'),
            'rolling_test_mean': res.get('rolling_test_mean')
        }
        try:
            pd.Series(flat).to_csv(out_csv)
            log.info(f'Validation summary written to {out_csv}')
        except Exception:
            log.exception('Could not write out CSV; skipping')

    return res

# End of module


