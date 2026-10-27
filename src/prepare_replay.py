"""Prepare replay input/output CSVs using the saved Week 2 model bundle.
Run from repository root: python src/prepare_replay.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model_bundle, predict_transactions

DATA_PATH = ROOT / "data" / "test.csv"
BUNDLE_PATH = ROOT / "models" / "fraud_xgboost_pipeline.joblib"
OUT_DIR = ROOT / "reports" / "week3" / "replay"
REPLAY_SAMPLE_SIZE = None  # None = all test rows; coordinate with teammates
SORT_BY_TIME_IF_AVAILABLE = True

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
bundle = load_model_bundle(BUNDLE_PATH)
data = pd.read_csv(DATA_PATH)
if "Class" not in data.columns:
    actual_available = False
else:
    actual_available = True
features = list(bundle["feature_columns"])
missing = [c for c in features if c not in data.columns]
if missing:
    raise ValueError("Missing model feature(s) in replay data: " + ", ".join(missing))

data = data.copy()
data["source_row"] = np.arange(len(data))
data["transaction_id"] = [f"test-row-{i:06d}" for i in range(len(data))]
if SORT_BY_TIME_IF_AVAILABLE and "Time" in data.columns:
    data = data.sort_values("Time", kind="mergesort").reset_index(drop=True)
    order_description = "ascending Time within data/test.csv"
else:
    data = data.reset_index(drop=True)
    order_description = "source CSV row order"
if REPLAY_SAMPLE_SIZE is not None:
    if not isinstance(REPLAY_SAMPLE_SIZE, int) or REPLAY_SAMPLE_SIZE < 1:
        raise ValueError("REPLAY_SAMPLE_SIZE must be a positive integer or None")
    data = data.head(REPLAY_SAMPLE_SIZE).copy()
data["replay_sequence"] = np.arange(1, len(data) + 1)

# Class is excluded: input contract contains feature columns plus replay metadata only.
inputs = data[["replay_sequence", "transaction_id"] + features].copy()
OUT_DIR.mkdir(parents=True, exist_ok=True)
input_path = OUT_DIR / "replay_inputs.csv"
output_path = OUT_DIR / "replay_outputs.csv"
manifest_path = OUT_DIR / "replay_manifest.json"
inputs.to_csv(input_path, index=False)

predictions = predict_transactions(inputs, BUNDLE_PATH)
# Ground-truth labels are attached after inference for evaluation only.
if actual_available:
    predictions["actual_label"] = data["Class"].astype(int).to_numpy()
if "Amount" in data.columns:
    predictions["Amount"] = data["Amount"].to_numpy()
columns = ["replay_sequence", "transaction_id", "actual_label", "Amount",
           "fraud_probability", "predicted_label", "predicted_class", "threshold_used"]
predictions = predictions[[c for c in columns if c in predictions.columns]]
predictions.to_csv(output_path, index=False)
manifest = {
    "source": "data/test.csv",
    "model_name": bundle.get("model_name", "selected_model"),
    "model_artifact": "models/fraud_xgboost_pipeline.joblib",
    "threshold": float(bundle["threshold"]),
    "target_mapping": {"0": "Legitimate", "1": "Fraud"},
    "feature_columns_in_order": features,
    "rows": int(len(inputs)),
    "replay_order": order_description,
    "actual_labels_in_input": False,
    "actual_labels_attached_after_prediction": actual_available,
    "input_file": "reports/week3/replay/replay_inputs.csv",
    "output_file": "reports/week3/replay/replay_outputs.csv",
    "warning": "test split replay is for offline verification, not proof of production drift. Never tune on replay/test labels."
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Replay completed.")
print("Input:", input_path)
print("Output:", output_path)
print("Manifest:", manifest_path)
print("Rows:", len(inputs), "Threshold:", bundle["threshold"])
print(predictions.head(10).to_string(index=False))
