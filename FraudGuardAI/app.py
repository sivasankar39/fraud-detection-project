
from flask import Flask, render_template, jsonify, request
from pathlib import Path
import os

import pandas as pd
import numpy as np

import joblib
from functools import lru_cache


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
PR_RESULTS_PATH = os.path.join(BASE_DIR, "pr_results.csv")

# AI/ML project files are in the parent project folder
PROJECT_ROOT = Path(BASE_DIR).parent

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard_intermediate.csv")
VALIDATION_PATH = PROJECT_ROOT / "data" / "validation.csv"
MODEL_BUNDLE_PATH = (
    PROJECT_ROOT / "models" / "fraud_xgboost_pipeline.joblib"
)


# ==================================================
# LOAD DATASET
# ==================================================

def load_data():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

        required_columns = {"Class", "Amount"}

        if not required_columns.issubset(df.columns):
            raise ValueError(
                "Dataset must contain Class and Amount columns."
            )

        df["Class"] = pd.to_numeric(
            df["Class"], errors="coerce"
        )

        df["Amount"] = pd.to_numeric(
            df["Amount"], errors="coerce"
        )

        df = df.dropna(
            subset=["Class", "Amount"]
        ).copy()

        df["Class"] = df["Class"].astype(int)

        if "Time" in df.columns:
            df["Time"] = pd.to_numeric(
                df["Time"], errors="coerce"
            )

        return df

    # Testing fallback only
    return pd.DataFrame({
        "Time": [100, 200, 300, 400, 500, 600],
        "Amount": [20, 150, 1200, 50, 800, 300],
        "Class": [0, 0, 1, 0, 1, 0]
    })


# ==================================================
# DASHBOARD CHART DATA
# ==================================================

def create_chart_data(df):
    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    class_chart = {
        "labels": ["Fraudulent", "Legitimate"],
        "values": [fraud_count, legitimate_count]
    }

    amount_edges = [
        -0.001, 1, 2, 5, 10, 20, 50, 100,
        200, 500, 1000, 2500, 5000,
        10000, np.inf
    ]

    amount_labels = [
        "0–1", "1–2", "2–5", "5–10", "10–20",
        "20–50", "50–100", "100–200",
        "200–500", "500–1K", "1K–2.5K",
        "2.5K–5K", "5K–10K", "10K+"
    ]

    amount_bins = pd.cut(
        df["Amount"],
        bins=amount_edges,
        labels=amount_labels,
        right=True,
        include_lowest=True
    )

    amount_table = pd.crosstab(
        amount_bins,
        df["Class"]
    ).reindex(
        index=amount_labels,
        columns=[0, 1],
        fill_value=0
    )

    amount_chart = {
        "labels": amount_labels,
        "legitimate": amount_table[0].astype(int).tolist(),
        "fraud": amount_table[1].astype(int).tolist()
    }

    if "Time" in df.columns and df["Time"].notna().any():
        timed_df = df.dropna(subset=["Time"]).copy()

        timed_df["ElapsedHour"] = (
            timed_df["Time"] // 3600
        ).astype(int)

        time_table = pd.crosstab(
            timed_df["ElapsedHour"],
            timed_df["Class"]
        )

        all_hours = list(
            range(int(timed_df["ElapsedHour"].max()) + 1)
        )

        time_table = time_table.reindex(
            index=all_hours,
            columns=[0, 1],
            fill_value=0
        )

        time_chart = {
            "labels": [str(hour) for hour in all_hours],
            "legitimate": time_table[0].astype(int).tolist(),
            "fraud": time_table[1].astype(int).tolist()
        }
    else:
        time_chart = {
            "labels": [],
            "legitimate": [],
            "fraud": []
        }

    return {
        "class_chart": class_chart,
        "amount_chart": amount_chart,
        "time_chart": time_chart
    }


# ==================================================
# DASHBOARD STATISTICS
# ==================================================

def get_stats(df):
    total = len(df)
    fraud = int((df["Class"] == 1).sum())
    legitimate = int((df["Class"] == 0).sum())

    fraud_rate = (
        fraud / total * 100
        if total else 0
    )

    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": round(fraud_rate, 3),
        "total_amount": round(
            float(df["Amount"].sum()), 2
        )
    }


# ==================================================
# OVERVIEW PAGE
# ==================================================

@app.route("/")
def overview():
    df = load_data()

    return render_template(
        "index.html",
        stats=get_stats(df),
        chart_data=create_chart_data(df)
    )


# ==============================================
# ADVANCED FRAUD ANALYTICS
# ==============================================

@app.route("/analytics")
def analytics():
    df = load_data()
    fraud_df = df[df["Class"] == 1].copy()

    # Detailed amount ranges — old dashboard style
    edges = [
        0, 2, 5, 10, 20, 50, 100,
        200, 500, 1000, 2000, 5000,
        float("inf")
    ]

    labels = [
        "0–2", "2–5", "5–10", "10–20",
        "20–50", "50–100", "100–200",
        "200–500", "500–1K", "1K–2K",
        "2K–5K", "5K+"
    ]

    all_bands = pd.cut(
        df["Amount"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True
    )

    fraud_bands = pd.cut(
        fraud_df["Amount"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True
    )

    total_counts = (
        all_bands.value_counts()
        .reindex(labels, fill_value=0)
        .astype(int)
    )

    fraud_counts = (
        fraud_bands.value_counts()
        .reindex(labels, fill_value=0)
        .astype(int)
    )

    fraud_rates = [
        round(100 * int(f) / int(t), 4)
        if int(t) > 0 else None
        for f, t in zip(
            fraud_counts.tolist(),
            total_counts.tolist()
        )
    ]

    # Hourly transaction analysis
    valid_df = df.dropna(subset=["Time"])
    valid_fraud = fraud_df.dropna(subset=["Time"])

    all_hours = (
        valid_df["Time"] // 3600
    ).astype(int)

    fraud_hours = (
        valid_fraud["Time"] // 3600
    ).astype(int)

    hours = sorted(all_hours.unique().tolist())

    hourly_total = (
        all_hours.value_counts()
        .reindex(hours, fill_value=0)
        .astype(int)
    )

    hourly_fraud = (
        fraud_hours.value_counts()
        .reindex(hours, fill_value=0)
        .astype(int)
    )

    hourly_rates = [
        round(100 * int(f) / int(t), 4)
        if int(t) > 0 else None
        for f, t in zip(
            hourly_fraud.tolist(),
            hourly_total.tolist()
        )
    ]

    # Actual labeled fraud transactions
    transactions = [
        {
            "id": int(idx) + 1,
            "time": (
                float(row["Time"])
                if pd.notna(row["Time"])
                else None
            ),
            "amount": float(row["Amount"]),
            "class": 1
        }
        for idx, row in fraud_df.iterrows()
    ]

    amounts = fraud_df["Amount"]

    analytics_data = {
        "summary": {
            "mean": float(amounts.mean()),
            "median": float(amounts.median()),
            "maximum": float(amounts.max()),
            "total": float(amounts.sum())
        },
        "amount_bands": {
            "labels": labels,
            "fraud_counts": fraud_counts.tolist(),
            "total_counts": total_counts.tolist(),
            "fraud_rates": fraud_rates
        },
        "hourly": {
            "labels": [str(h) for h in hours],
            "fraud_counts": hourly_fraud.tolist(),
            "total_counts": hourly_total.tolist(),
            "fraud_rates": hourly_rates
        },
        "fraud_transactions": transactions
    }

    return render_template(
        "analytics.html",
        stats=get_stats(df),
        analytics_data=analytics_data
    )


# ==================================================
# COST OPTIMIZER PAGE
# ==================================================

@lru_cache(maxsize=1)
def load_cost_optimizer_data():
    """Load the saved model and validation predictions once."""

    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Trained model bundle not found: {MODEL_BUNDLE_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    feature_columns = bundle["feature_columns"]

    required_columns = set(feature_columns) | {"Class", "Amount"}
    missing_columns = required_columns - set(validation_df.columns)

    if missing_columns:
        raise ValueError(
            f"Validation dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    X_val = validation_df[feature_columns]
    y_true = validation_df["Class"].to_numpy(dtype=int)
    amounts = validation_df["Amount"].to_numpy(dtype=float)

    probabilities = bundle["model"].predict_proba(X_val)[:, 1]

    return {
        "bundle": bundle,
        "y_true": y_true,
        "amounts": amounts,
        "probabilities": probabilities,
    }


def calculate_optimizer_metrics(fp_unit_cost, threshold, fn_multiplier=1.0):
    """Calculate validation-set costs for the selected assumptions."""

    data = load_cost_optimizer_data()

    y_true = data["y_true"]
    amounts = data["amounts"]
    probabilities = data["probabilities"]

    predictions = (probabilities >= threshold).astype(int)

    fp_mask = (y_true == 0) & (predictions == 1)
    fn_mask = (y_true == 1) & (predictions == 0)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())

    fp_cost = fp_count * fp_unit_cost

    # Baseline missed-fraud loss is the sum of the missed fraud Amounts.
    # The multiplier allows scenario analysis; 1.0 means the original
    # notebook assumption is used without adjustment.
    fn_loss = float(amounts[fn_mask].sum()) * fn_multiplier

    total_cost = fp_cost + fn_loss

    actual_fraud = int((y_true == 1).sum())
    detected_fraud = int(((y_true == 1) & (predictions == 1)).sum())

    precision = (
        float(((y_true == 1) & (predictions == 1)).sum())
        / max(int(predictions.sum()), 1)
    )

    recall = detected_fraud / max(actual_fraud, 1)

    return {
        "validation_transactions": int(len(y_true)),
        "actual_fraud": actual_fraud,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "fp_unit_cost": float(fp_unit_cost),
        "fp_cost": float(fp_cost),
        "fn_loss": float(fn_loss),
        "total_cost": float(total_cost),
        "precision": precision,
        "recall": recall,
        "threshold": float(threshold),
        "fn_multiplier": float(fn_multiplier),
    }

@app.route("/cost-optimizer")
def cost_optimizer():
    df = load_data()
    model_data = load_cost_optimizer_data()

    return render_template(
        "cost_optimizer.html",
        stats=get_stats(df),
        default_threshold=float(
            model_data["bundle"].get("threshold", 0.78)
        ),
    )


@app.route("/api/cost-optimizer", methods=["POST"])
def cost_optimizer_api():
    try:
        payload = request.get_json(silent=True) or {}

        fp_cost = float(payload.get("fp_cost", 2))
        threshold = float(payload.get("threshold", 0.78))
        fn_multiplier = float(payload.get("fn_multiplier", 1.0))

        if not 0 <= fp_cost <= 50:
            return jsonify({"error": "FP cost must be between $0 and $50."}), 400

        if not 0 <= threshold <= 1:
            return jsonify({"error": "Threshold must be between 0 and 1."}), 400

        if not 0 <= fn_multiplier <= 2:
            return jsonify({"error": "Loss multiplier must be between 0 and 2."}), 400

        return jsonify(
            calculate_optimizer_metrics(
                fp_unit_cost=fp_cost,
                threshold=threshold,
                fn_multiplier=fn_multiplier,
            )
        )

    except Exception as exc:
        app.logger.exception("Cost optimizer calculation failed")
        return jsonify({"error": str(exc)}), 500


# ==================================================
# PRECISION-RECALL EXPLORER
# ==================================================

@app.route("/pr-explorer")
def pr_explorer():
    df = load_data()

    pr_data = {
        "precision": [],
        "recall": [],
        "average_precision": None,
        "status": "Model results not available.",
        "model_available": MODEL_BUNDLE_PATH.exists()
    }

    try:
        from sklearn.metrics import (
            precision_recall_curve,
            average_precision_score,
        )

        analysis = load_cost_optimizer_data()

        y_true = analysis["y_true"]
        probabilities = analysis["probabilities"]

        precision, recall, _ = precision_recall_curve(
            y_true,
            probabilities
        )

        average_precision = average_precision_score(
            y_true,
            probabilities
        )

        pr_data["precision"] = precision.tolist()
        pr_data["recall"] = recall.tolist()
        pr_data["average_precision"] = round(
            float(average_precision), 6
        )
        pr_data["status"] = (
            "Precision-Recall curve generated using the saved "
            "model and validation dataset."
        )

    except (FileNotFoundError, ValueError, KeyError, OSError):
        app.logger.exception(
            "Could not generate Precision-Recall results"
        )
        pr_data["status"] = (
            "Could not generate results. Check the model bundle "
            "and validation dataset."
        )

    return render_template(
        "pr_explorer.html",
        stats=get_stats(df),
        pr_data=pr_data
    )



# ==================================================
# LIVE SIMULATOR PAGE
# ==================================================

@app.route("/live-simulator")
def live_simulator():
    df = load_data()

    return render_template(
        "live_simulator.html",
        stats=get_stats(df)
    )


# ==================================================
# LIVE SIMULATOR TRANSACTION API
# ==================================================


@app.route("/api/simulator/transactions")
def simulator_transactions():
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = max(1, min(int(request.args.get("limit", 10)), 100))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid offset or limit."}), 400

    try:
        df = load_data().copy()

        if not MODEL_BUNDLE_PATH.exists():
            return jsonify({"error": "Trained model bundle not found."}), 500

        # Load the trained model and its expected feature columns
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]

        required_columns = {"Time", "Amount", "Class"} | set(feature_columns)
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            return jsonify({
                "error": "Dataset is missing required model features.",
                "missing_columns": sorted(missing_columns)
            }), 500

        df["transaction_id"] = df.index + 1

        df = df.dropna(subset=list(required_columns))
        df = df.sort_values(
            by=["Time", "transaction_id"],
            ascending=[True, True],
            kind="stable"
        )

        total = len(df)
        batch = df.iloc[offset:offset + limit].copy()

        # Generate model predictions for this batch
        if not batch.empty:
            X_batch = batch[feature_columns]
            probabilities = model.predict_proba(X_batch)[:, 1]
            predicted_classes = (probabilities >= 0.5).astype(int)
        else:
            probabilities = []
            predicted_classes = []

        transactions = []

        for i, (_, row) in enumerate(batch.iterrows()):
            actual_class = int(row["Class"])
            predicted_class = int(predicted_classes[i])
            probability = float(probabilities[i])

            transactions.append({
                "id": int(row["transaction_id"]),
                "time": float(row["Time"]),
                "amount": round(float(row["Amount"]), 2),

                "actual_class": actual_class,
                "actual_label": (
                    "Fraud" if actual_class == 1 else "Legitimate"
                ),

                "predicted_class": predicted_class,
                "predicted_label": (
                    "Fraud" if predicted_class == 1 else "Legitimate"
                ),
                "fraud_probability": round(probability, 6)
            })

        next_offset = offset + len(transactions)

        return jsonify({
            "transactions": transactions,
            "total": total,
            "next_offset": next_offset,
            "has_more": next_offset < total,
            "mode": "historical_replay_with_model_predictions"
        })

    except Exception:
        app.logger.exception("Simulator API failed")
        return jsonify({
            "error": "Could not generate simulator predictions. Check the model bundle and server logs."
        }), 500


# ==================================================
# INVESTIGATION QUEUE
# ==================================================

@app.route("/investigation-queue")
def investigation_queue():
    df = load_data()

    return render_template(
        "investigation_queue.html",
        stats=get_stats(df)
    )


@app.route("/api/investigation-queue")
def investigation_queue_api():
    try:
        df = load_data().copy()

        required = {"Time", "Amount", "Class"}

        if not required.issubset(df.columns):
            return jsonify({
                "error": "Required dataset columns missing."
            }), 400

        df["transaction_id"] = df.index + 1

        df = df.dropna(
            subset=["Time", "Amount", "Class"]
        )

        # Demo queue uses actual labeled fraud cases.
        fraud_df = df[df["Class"] == 1].copy()

        fraud_df = fraud_df.sort_values(
            "Amount",
            ascending=False
        )

        cases = []

        for row in fraud_df.itertuples():
            amount = float(row.Amount)

            # Demo triage heuristic, not ML risk score.
            if amount >= 500:
                priority = "Critical"
            elif amount >= 100:
                priority = "High"
            elif amount >= 20:
                priority = "Medium"
            else:
                priority = "Low"

            cases.append({
                "id": int(row.transaction_id),
                "time": round(float(row.Time), 0),
                "amount": round(amount, 2),
                "priority": priority,
                "actual_label": "Fraud"
            })

        return jsonify({
            "cases": cases,
            "total": len(cases),
            "mode": "ground_truth_demo"
        })

    except Exception:
        app.logger.exception("Investigation queue failed")

        return jsonify({
            "error": "Unable to load investigation cases."
        }), 500

# ==================================================
# DRIFT MONITOR
# ==================================================

@app.route("/drift-monitor")
def drift_monitor():
    df = load_data()
    return render_template(
        "drift_monitor.html",
        stats=get_stats(df)
    )


@app.route("/api/drift-monitor")
def drift_monitor_api():
    try:
        window = int(request.args.get("window", 10000))
        if window < 100 or window > 100000:
            return jsonify({
                "error": "Window must be between 100 and 100000."
            }), 400

        df = load_data().copy()

        required = {"Time", "Amount", "Class"}
        if not required.issubset(df.columns):
            return jsonify({
                "error": "Missing Time, Amount or Class."
            }), 400

        for column in required:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df = df.dropna(subset=list(required))
        df = df.sort_values("Time", kind="stable")

        if len(df) < 2 * window:
            return jsonify({
                "error": "Not enough transactions for two windows."
            }), 400

        reference = df.iloc[:window]
        current = df.iloc[-window:]

        ref_amount = reference["Amount"].to_numpy()
        cur_amount = current["Amount"].to_numpy()

        # Fixed business-readable amount bands
        # used for charts, not PSI binning.
        chart_edges = [
            -np.inf, 1, 10, 50, 100,
            500, 1000, np.inf
        ]

        chart_labels = [
            "0–1", "1–10", "10–50", "50–100",
            "100–500", "500–1K", "1K+"
        ]

        ref_counts, _ = np.histogram(
            ref_amount, bins=chart_edges
        )
        cur_counts, _ = np.histogram(
            cur_amount, bins=chart_edges
        )

        ref_pct = ref_counts / len(ref_amount) * 100
        cur_pct = cur_counts / len(cur_amount) * 100

        # PSI bins from reference distribution only.
        # Deduplicate quantiles to handle repeated amounts.
        quantiles = np.quantile(
            ref_amount, np.linspace(0, 1, 11)
        )
        inner_edges = np.unique(quantiles[1:-1])
        psi_edges = np.concatenate((
            [-np.inf], inner_edges, [np.inf]
        ))

        ref_psi_counts, _ = np.histogram(
            ref_amount, bins=psi_edges
        )
        cur_psi_counts, _ = np.histogram(
            cur_amount, bins=psi_edges
        )

        # Additive smoothing prevents log(0).
        bins_count = len(ref_psi_counts)
        alpha = 0.5

        ref_share = (
            (ref_psi_counts + alpha)
            / (len(ref_amount) + alpha * bins_count)
        )
        cur_share = (
            (cur_psi_counts + alpha)
            / (len(cur_amount) + alpha * bins_count)
        )

        psi = float(np.sum(
            (cur_share - ref_share)
            * np.log(cur_share / ref_share)
        ))

        # Conventional heuristic thresholds
        if psi < 0.1:
            drift_status = "Stable"
        elif psi < 0.25:
            drift_status = "Moderate Drift"
        else:
            drift_status = "Significant Drift"

        ref_fraud = int(reference["Class"].eq(1).sum())
        cur_fraud = int(current["Class"].eq(1).sum())

        ref_fraud_rate = ref_fraud / window * 100
        cur_fraud_rate = cur_fraud / window * 100

        # Relative elapsed-hour fraud pattern
        def hourly_pattern(frame):
            hours = (
                frame["Time"] // 3600
            ).astype(int)

            grouped = frame.groupby(hours)["Class"].agg(
                ["sum", "count"]
            )

            return grouped

        ref_hours = hourly_pattern(reference)
        cur_hours = hourly_pattern(current)

        all_hours = sorted(
            set(ref_hours.index) | set(cur_hours.index)
        )

        def hourly_rates(grouped):
            rates = (
                grouped["sum"] / grouped["count"] * 100
            ).reindex(all_hours).fillna(0)
            return rates.round(4).tolist()

        return jsonify({
            "window": window,
            "reference": {
                "count": len(reference),
                "fraud_count": ref_fraud,
                "fraud_rate": round(ref_fraud_rate, 4),
                "average_amount": round(
                    float(reference["Amount"].mean()), 2
                )
            },
            "current": {
                "count": len(current),
                "fraud_count": cur_fraud,
                "fraud_rate": round(cur_fraud_rate, 4),
                "average_amount": round(
                    float(current["Amount"].mean()), 2
                )
            },
            "psi": round(psi, 5),
            "drift_status": drift_status,
            "amount_distribution": {
                "labels": chart_labels,
                "reference": np.round(
                    ref_pct, 3
                ).tolist(),
                "current": np.round(
                    cur_pct, 3
                ).tolist()
            },
            "hourly_pattern": {
                "labels": [
                    str(h) for h in all_hours
                ],
                "reference": hourly_rates(ref_hours),
                "current": hourly_rates(cur_hours)
            },
            "mode": "historical_window_comparison"
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid window parameter."
        }), 400
    except Exception:
        app.logger.exception("Drift monitoring failed")
        return jsonify({
            "error": "Could not calculate drift."
        }), 500

@app.route("/api/analytics/fraud-transactions")
def analytics_fraud_transactions():
    try:
        df = load_data()

        if "Class" not in df.columns:
            return jsonify({
                "error": "Class column missing from dataset"
            }), 500

        fraud_df = df[df["Class"] == 1]

        records = []

        for index, row in fraud_df.iterrows():
            records.append({
                "id": int(index) + 1,
                "time": (
                    float(row["Time"])
                    if pd.notna(row["Time"])
                    else None
                ),
                "amount": (
                    float(row["Amount"])
                    if pd.notna(row["Amount"])
                    else None
                ),
                "class": 1
            })

        return jsonify({
            "success": True,
            "total": len(records),
            "transactions": records
        })

    except Exception as error:
        app.logger.exception(
            "Failed to load fraud transactions"
        )

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


# ==================================================
# RUN FLASK APPLICATION
# KEEP THIS BLOCK AT THE VERY END
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)
