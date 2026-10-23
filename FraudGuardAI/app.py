
from flask import Flask, render_template, jsonify, request
from pathlib import Path
import os

import pandas as pd
import numpy as np


# ==================================================
# FLASK APPLICATION
# ==================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
PR_RESULTS_PATH = os.path.join(BASE_DIR, "pr_results.csv")


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

@app.route("/cost-optimizer")
def cost_optimizer():
    df = load_data()

    return render_template(
        "cost_optimizer.html",
        stats=get_stats(df)
    )


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
        "status": "Model results not found.",
        "model_available": os.path.exists(MODEL_PATH)
    }

    if os.path.exists(PR_RESULTS_PATH):
        try:
            results = pd.read_csv(PR_RESULTS_PATH)

            required_columns = {"precision", "recall"}

            if required_columns.issubset(results.columns):
                results["precision"] = pd.to_numeric(
                    results["precision"],
                    errors="coerce"
                )

                results["recall"] = pd.to_numeric(
                    results["recall"],
                    errors="coerce"
                )

                results = results.dropna(
                    subset=["precision", "recall"]
                )

                results = results[
                    results["precision"].between(0, 1)
                    & results["recall"].between(0, 1)
                ].copy()

                if not results.empty:
                    pr_data["precision"] = (
                        results["precision"].tolist()
                    )

                    pr_data["recall"] = (
                        results["recall"].tolist()
                    )

                    recall = results["recall"].to_numpy()
                    precision = results["precision"].to_numpy()

                    complete_curve = (
                        len(recall) >= 2
                        and abs(recall[0] - 1.0) < 1e-8
                        and abs(recall[-1]) < 1e-8
                        and (
                            recall[:-1] >= recall[1:]
                        ).all()
                    )

                    if complete_curve:
                        ap = -np.sum(
                            np.diff(recall) * precision[:-1]
                        )

                        pr_data["average_precision"] = (
                            round(float(ap), 6)
                        )

                        pr_data["status"] = (
                            "Precision-Recall results loaded. "
                            "Average Precision calculated."
                        )
                    else:
                        pr_data["status"] = (
                            "PR curve loaded. Average Precision "
                            "requires a complete ordered curve."
                        )
                else:
                    pr_data["status"] = (
                        "No valid Precision-Recall values found."
                    )
            else:
                pr_data["status"] = (
                    "CSV must contain precision and recall columns."
                )

        except (OSError, pd.errors.ParserError, ValueError):
            app.logger.exception(
                "Could not read PR results"
            )

            pr_data["status"] = (
                "Could not read PR results CSV."
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
        offset = int(
            request.args.get("offset", 0)
        )
        limit = int(
            request.args.get("limit", 10)
        )

        offset = max(0, offset)
        limit = max(1, min(limit, 100))

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid offset or limit."
        }), 400

    try:
        df = load_data().copy()

        required_columns = {
            "Time",
            "Amount",
            "Class"
        }

        if not required_columns.issubset(df.columns):
            return jsonify({
                "error": (
                    "Dataset must contain "
                    "Time, Amount and Class."
                )
            }), 400

        df["transaction_id"] = df.index + 1

        df = df.dropna(
            subset=["Time", "Amount", "Class"]
        )

        df = df.sort_values(
            by=["Time", "transaction_id"],
            ascending=[True, True],
            kind="stable"
        )

        total = len(df)

        batch = df.iloc[
            offset:offset + limit
        ]

        transactions = []

        for row in batch.itertuples():
            actual_class = int(row.Class)

            transactions.append({
                "id": int(row.transaction_id),
                "time": float(row.Time),
                "amount": round(float(row.Amount), 2),
                "actual_class": actual_class,
                "actual_label": (
                    "Fraud"
                    if actual_class == 1
                    else "Legitimate"
                )
            })

        next_offset = offset + len(transactions)

        return jsonify({
            "transactions": transactions,
            "total": total,
            "next_offset": next_offset,
            "has_more": next_offset < total,
            "mode": "historical_label_replay"
        })

    except Exception:
        app.logger.exception(
            "Simulator API failed"
        )

        return jsonify({
            "error": (
                "Could not load simulator transactions."
            )
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
