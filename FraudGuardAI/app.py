
from flask import Flask, render_template
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
PR_RESULTS_PATH = os.path.join(BASE_DIR, "pr_results.csv")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
def load_data():
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

        required_columns = {"Class", "Amount"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                "Dataset must contain 'Class' and 'Amount' columns."
            )

        df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

        df = df.dropna(subset=["Class", "Amount"]).copy()
        df["Class"] = df["Class"].astype(int)

        if "Time" in df.columns:
            df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

        return df

    # Fallback sample data for testing only
    return pd.DataFrame({
        "Time": [100, 200, 300, 400, 500, 600],
        "Amount": [20, 150, 1200, 50, 800, 300],
        "Class": [0, 0, 1, 0, 1, 0]
    })


# --------------------------------------------------
# DASHBOARD CHART DATA
# --------------------------------------------------
def create_chart_data(df):
    fraud_count = int((df["Class"] == 1).sum())
    legitimate_count = int((df["Class"] == 0).sum())

    class_chart = {
        "labels": ["Fraudulent", "Legitimate"],
        "values": [fraud_count, legitimate_count]
    }

    amount_edges = [
        -0.001, 1, 2, 5, 10, 20, 50, 100,
        200, 500, 1000, 2500, 5000, 10000, np.inf
    ]

    amount_labels = [
        "0–1", "1–2", "2–5", "5–10", "10–20",
        "20–50", "50–100", "100–200", "200–500",
        "500–1K", "1K–2.5K", "2.5K–5K",
        "5K–10K", "10K+"
    ]

    amount_bins = pd.cut(
        df["Amount"],
        bins=amount_edges,
        labels=amount_labels,
        right=True,
        include_lowest=True
    )

    amount_table = pd.crosstab(
        amount_bins, df["Class"]
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


# --------------------------------------------------
# DASHBOARD STATISTICS
# --------------------------------------------------
def get_stats(df):
    total = len(df)
    fraud = int((df["Class"] == 1).sum())
    legitimate = int((df["Class"] == 0).sum())

    fraud_rate = fraud / total * 100 if total else 0

    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": round(fraud_rate, 3),
        "total_amount": round(float(df["Amount"].sum()), 2)
    }


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------
@app.route("/")
def overview():
    df = load_data()

    return render_template(
        "index.html",
        stats=get_stats(df),
        chart_data=create_chart_data(df)
    )


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------
@app.route("/analytics")
def analytics():
    df = load_data()

    fraud_df = df[df["Class"] == 1].copy()
    fraud_rows = []

    for index, row in fraud_df.iterrows():
        fraud_rows.append({
            "transaction_id": int(index) + 1,
            "time": (
                round(float(row["Time"]), 2)
                if "Time" in df.columns
                and pd.notna(row.get("Time"))
                else None
            ),
            "amount": round(float(row["Amount"]), 2)
        })

    return render_template(
        "analytics.html",
        stats=get_stats(df),
        chart_data=create_chart_data(df),
        fraud_rows=fraud_rows
    )


# --------------------------------------------------
# COST OPTIMIZER
# --------------------------------------------------
@app.route("/cost-optimizer")
def cost_optimizer():
    df = load_data()

    return render_template(
        "cost_optimizer.html",
        stats=get_stats(df)
    )


# --------------------------------------------------
# PRECISION-RECALL EXPLORER
# --------------------------------------------------
@app.route("/pr-explorer")
def pr_explorer():
    df = load_data()

    pr_data = {
        "precision": [],
        "recall": [],
        "average_precision": None,
        "status": "Model results not found."
    }

    if os.path.exists(PR_RESULTS_PATH):
        try:
            results = pd.read_csv(PR_RESULTS_PATH)

            required_columns = {"precision", "recall"}

            if required_columns.issubset(results.columns):
                results["precision"] = pd.to_numeric(
                    results["precision"], errors="coerce"
                )
                results["recall"] = pd.to_numeric(
                    results["recall"], errors="coerce"
                )

                results = results.dropna(
                    subset=["precision", "recall"]
                )

                pr_data["precision"] = (
                    results["precision"].tolist()
                )
                pr_data["recall"] = (
                    results["recall"].tolist()
                )

                if len(results) > 0:
                    pr_data["status"] = (
                        "Precision-Recall results loaded."
                    )

            else:
                pr_data["status"] = (
                    "CSV must contain precision and recall columns."
                )

        except (OSError, pd.errors.ParserError, ValueError) as error:
            app.logger.exception("Could not read PR results")
            pr_data["status"] = "Could not read PR results CSV."

    if os.path.exists(MODEL_PATH):
        pr_data["model_available"] = True
    else:
        pr_data["model_available"] = False

    return render_template(
        "pr_explorer.html",
        stats=get_stats(df),
        pr_data=pr_data
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
