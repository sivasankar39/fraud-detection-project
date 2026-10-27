"""End-to-end smoke checks for W3-ML-01, W3-ML-02 and selected MVP behavior.
Run from repository root after packaging and replay:
    python src/run_week3_checks.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import load_model_bundle, predict_transactions
from src.week3_monitoring import calculate_business_cost, drift_summary

BUNDLE = ROOT / "models" / "fraud_xgboost_pipeline.joblib"
TEST_DATA = ROOT / "data" / "test.csv"
TRAIN_DATA = ROOT / "data" / "train.csv"
REPLAY_DIR = ROOT / "reports" / "week3" / "replay"
INPUT_PATH = REPLAY_DIR / "replay_inputs.csv"
OUTPUT_PATH = REPLAY_DIR / "replay_outputs.csv"
MANIFEST_PATH = REPLAY_DIR / "replay_manifest.json"

checks = []
def check(name, condition, detail=""):
    ok = bool(condition)
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

# W3-ML-01: artifact, schema, probability and threshold consistency.
try:
    bundle = load_model_bundle(BUNDLE)
    check("Saved model bundle loads", True, str(BUNDLE))
except Exception as exc:
    check("Saved model bundle loads", False, str(exc))
    print("\nPackaging must be completed before the remaining integration checks.")
    sys.exit(1)

features = list(bundle["feature_columns"])
threshold = float(bundle["threshold"])
check("Threshold is between 0 and 1", 0 <= threshold <= 1, str(threshold))
check("Feature schema is non-empty", len(features) > 0, f"{len(features)} features")

if not TEST_DATA.exists():
    check("Test data exists", False, str(TEST_DATA))
    sys.exit(1)
test = pd.read_csv(TEST_DATA)
missing = [f for f in features if f not in test.columns]
check("Test data contains every feature", not missing, f"missing={missing}")
if missing:
    sys.exit(1)

sample = test.head(min(20, len(test))).copy()
pred = predict_transactions(sample, BUNDLE)
prob = pred["fraud_probability"].to_numpy()
labels = pred["predicted_label"].to_numpy()
check("Prediction count matches input rows", len(pred) == len(sample), f"{len(pred)} rows")
check("Probabilities are finite and within [0, 1]", np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all())
check("Predictions match packaged threshold", np.array_equal(labels, (prob >= threshold).astype(int)))

# W3-ML-02: replay schema and output alignment.
required_paths = [INPUT_PATH, OUTPUT_PATH, MANIFEST_PATH]
for path in required_paths:
    check(f"Replay artifact exists: {path.name}", path.exists(), str(path))
if all(p.exists() for p in required_paths):
    replay_in = pd.read_csv(INPUT_PATH)
    replay_out = pd.read_csv(OUTPUT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    check("Replay input excludes target label", "Class" not in replay_in.columns and "actual_label" not in replay_in.columns)
    check("Replay input contains all model features", all(f in replay_in.columns for f in features))
    check("Replay input/output row counts align", len(replay_in) == len(replay_out) == manifest["rows"])
    check("Replay output has probability and label columns", {"fraud_probability", "predicted_label", "threshold_used"}.issubset(replay_out.columns))
    if len(replay_out):
        out_prob = replay_out["fraud_probability"].to_numpy()
        out_label = replay_out["predicted_label"].to_numpy()
        check("Replay probabilities are valid", np.isfinite(out_prob).all() and ((out_prob >= 0) & (out_prob <= 1)).all())
        check("Replay labels follow packaged threshold", np.array_equal(out_label, (out_prob >= threshold).astype(int)))
        check("Replay sequence is unique and ordered", replay_out["replay_sequence"].is_unique and replay_out["replay_sequence"].tolist() == list(range(1, len(replay_out) + 1)))

# W3-SH-01: business-cost slider behavior (test underlying calculator at multiple slider values).
try:
    cost2 = calculate_business_cost([0, 0, 1, 1], [1, 0, 0, 1], [100, 20, 35, 10], fp_cost=2)
    cost5 = calculate_business_cost([0, 0, 1, 1], [1, 0, 0, 1], [100, 20, 35, 10], fp_cost=5)
    expected_change = 3 * cost2["fp"]
    check("Cost slider calculation responds to FP cost", np.isclose(cost5["total_estimated_cost"] - cost2["total_estimated_cost"], expected_change), f"cost $2={cost2['total_estimated_cost']}, cost $5={cost5['total_estimated_cost']}")
    check("Cost output components sum correctly", np.isclose(cost2["total_estimated_cost"], cost2["false_positive_cost"] + cost2["estimated_missed_fraud_loss"]))
except Exception as exc:
    check("Cost calculation smoke test", False, str(exc))

# W3-SH-01: drift-summary calculation and output structure.
try:
    if TRAIN_DATA.exists():
        train_df = pd.read_csv(TRAIN_DATA, nrows=5000)
        current_df = test.head(5000)
        drift_columns = [c for c in features if c in train_df.columns and c in current_df.columns and pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(current_df[c])]
        drift = drift_summary(train_df, current_df, drift_columns[:20])
        check("Drift summary runs", isinstance(drift, pd.DataFrame), f"{len(drift)} features checked")
        check("Drift summary contains PSI and KS fields", {"feature", "psi", "ks_p_value", "psi_level"}.issubset(drift.columns))
        report_dir = ROOT / "reports" / "week3"
        report_dir.mkdir(parents=True, exist_ok=True)
        drift.to_csv(report_dir / "drift_smoke_test.csv", index=False)
        check("Drift smoke-test report saved", (report_dir / "drift_smoke_test.csv").exists())
    else:
        check("Drift summary runs", False, f"Missing {TRAIN_DATA}")
except Exception as exc:
    check("Drift summary runs", False, str(exc))

# Summary
failed = [name for name, ok, _ in checks if not ok]
print(f"\nChecks passed: {len(checks) - len(failed)}/{len(checks)}")
if failed:
    print("Failed checks:", ", ".join(failed))
    sys.exit(1)
print("All configured Week 3 smoke checks passed. This does not replace a manual UI/live integration review.")
