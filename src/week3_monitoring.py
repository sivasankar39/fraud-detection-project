"""Shared functions for business-cost controls and feature-drift summaries."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_business_cost(y_true, predictions, amounts, fp_cost=2.0):
    """Calculate illustrative cost: FP * slider cost + sum(Amount for FNs)."""
    if fp_cost < 0:
        raise ValueError("fp_cost must be >= 0")
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    amounts = np.asarray(amounts, dtype=float)
    if not (len(y_true) == len(predictions) == len(amounts)):
        raise ValueError("y_true, predictions and amounts must have equal lengths")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("y_true and predictions must contain only 0 or 1")
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("amounts must be finite and non-negative")

    fp_mask = (y_true == 0) & (predictions == 1)
    fn_mask = (y_true == 1) & (predictions == 0)
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp_cost_value = float(fp * fp_cost)
    fn_loss = float(amounts[fn_mask].sum())
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fp_cost_per_case": float(fp_cost),
        "false_positive_cost": fp_cost_value,
        "estimated_missed_fraud_loss": fn_loss,
        "total_estimated_cost": fp_cost_value + fn_loss,
    }


def calculate_psi(reference, current, bins=10, epsilon=1e-6):
    """Compute PSI with bin edges learned from reference data."""
    reference = pd.Series(reference).dropna().to_numpy(dtype=float)
    current = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # constant/near-constant reference feature
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    return float(np.sum((cur_pct - ref_pct) * np.log((cur_pct + epsilon) / (ref_pct + epsilon))))


def drift_summary(reference_df, current_df, columns=None):
    """Return a numeric feature drift table with PSI, KS and missing rates.

    PSI levels (<0.10 low, 0.10-0.20 moderate, >=0.20 high) are rough
    monitoring heuristics, not universal statistical cutoffs.
    """
    if columns is None:
        columns = [c for c in reference_df.columns if c in current_df.columns
                   and pd.api.types.is_numeric_dtype(reference_df[c])
                   and pd.api.types.is_numeric_dtype(current_df[c])]
    rows = []
    for column in columns:
        ref_all = reference_df[column]
        cur_all = current_df[column]
        ref = ref_all.dropna()
        cur = cur_all.dropna()
        if ref.empty or cur.empty:
            ks_stat, ks_p = np.nan, np.nan
        else:
            ks = ks_2samp(ref.to_numpy(), cur.to_numpy())
            ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
        psi = calculate_psi(ref, cur)
        if pd.isna(psi):
            level = "Not calculated"
        elif psi < 0.10:
            level = "Low"
        elif psi < 0.20:
            level = "Moderate"
        else:
            level = "High"
        rows.append({
            "feature": column,
            "reference_mean": float(ref.mean()) if len(ref) else np.nan,
            "current_mean": float(cur.mean()) if len(cur) else np.nan,
            "reference_missing_pct": float(ref_all.isna().mean() * 100),
            "current_missing_pct": float(cur_all.isna().mean() * 100),
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p,
            "psi": psi,
            "psi_level": level,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
    return result
