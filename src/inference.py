"""Load the saved Week 2 model and predict fraud probabilities/labels."""
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "models" / "fraud_xgboost_pipeline.joblib"


def load_model_bundle(bundle_path: Union[str, Path] = DEFAULT_BUNDLE_PATH) -> dict:
    """Load and validate the model pipeline, selected threshold, and feature schema."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {path}. Run the model-packaging cell first."
        )
    bundle = joblib.load(path)
    required = {"model", "threshold", "feature_columns"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
    if not hasattr(bundle["model"], "predict_proba"):
        raise TypeError("Saved model must provide predict_proba().")
    threshold = float(bundle["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid saved threshold: {threshold}")
    if not bundle["feature_columns"]:
        raise ValueError("feature_columns cannot be empty")
    return bundle


def predict_transactions(
    transactions: Union[pd.DataFrame, str, Path],
    bundle_path: Union[str, Path] = DEFAULT_BUNDLE_PATH,
) -> pd.DataFrame:
    """Return fraud probability, predicted label and saved threshold per row.

    Input may be a DataFrame or CSV path. The input must include all model
    features. Extra metadata/target columns are ignored because only the
    exact feature list from the model bundle is passed to the pipeline.
    """
    if isinstance(transactions, (str, Path)):
        data = pd.read_csv(transactions)
    elif isinstance(transactions, pd.DataFrame):
        data = transactions.copy()
    else:
        raise TypeError("transactions must be a pandas DataFrame or CSV path")

    bundle = load_model_bundle(bundle_path)
    features = list(bundle["feature_columns"])
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))

    # Preserve training-time feature order. The fitted model pipeline applies
    # its own preprocessing; do not fit or apply a scaler separately here.
    X = data.loc[:, features].copy()
    probabilities = np.asarray(bundle["model"].predict_proba(X)[:, 1], dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model produced invalid fraud probabilities")

    threshold = float(bundle["threshold"])
    predictions = (probabilities >= threshold).astype(int)
    output = pd.DataFrame(index=data.index)
    for column in ("replay_sequence", "transaction_id"):
        if column in data.columns:
            output[column] = data[column].to_numpy()
    output["fraud_probability"] = probabilities
    output["predicted_label"] = predictions
    output["predicted_class"] = np.where(predictions == 1, "Fraud", "Legitimate")
    output["threshold_used"] = threshold
    return output.reset_index(drop=True)
