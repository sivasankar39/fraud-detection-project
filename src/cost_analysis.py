
import pandas as pd
from sklearn.metrics import confusion_matrix

# Provisional assumption: cost of reviewing one false positive.
FP_COST = 2.0  # USD per false positive (illustrative assumption)


def calculate_business_cost(y_true, y_pred, amounts):
    """
    Calculate false-positive and false-negative costs.

    y_true: actual labels (0 = genuine, 1 = fraud)
    y_pred: predicted labels (0 = genuine, 1 = fraud)
    amounts: transaction amounts corresponding to each prediction
    """

    results = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "amount": amounts
    })

    # False positive: genuine transaction predicted as fraud.
    fp_mask = (
        (results["actual"] == 0) &
        (results["predicted"] == 1)
    )

    # False negative: fraudulent transaction predicted as genuine.
    fn_mask = (
        (results["actual"] == 1) &
        (results["predicted"] == 0)
    )

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())

    # FP cost is in USD, based on a provisional assumption.
    total_fp_cost = fp_count * FP_COST
    
    # FN loss is estimated using transaction amounts in USD.
    total_fn_loss = results.loc[fn_mask, "amount"].sum()

    print("False Positives:", fp_count)
    print("False Negatives:", fn_count)
    print(f"Estimated FP cost: ${total_fp_cost:.2f} (provisional)")
    print(f"Estimated FN loss: ${total_fn_loss:.2f} (estimated USD loss)")
    print(f"Estimated total cost: ${total_fp_cost + total_fn_loss:.2f} (USD)")


    return {
        "fp_count": fp_count,
        "fn_count": fn_count,
        "total_fp_cost": total_fp_cost,
        "total_fn_loss": total_fn_loss
    }



def calculate_cost_across_thresholds(
    y_true,
    y_prob,
    amounts,
    thresholds=None,
):
    """
    Calculate FP, FN and estimated total cost
    across multiple classification thresholds.

    y_true: Actual labels (0 = genuine, 1 = fraud)
    y_prob: Predicted fraud probabilities
    amounts: Transaction amounts in USD
    thresholds: Thresholds to evaluate
    """

    if thresholds is None:
        thresholds = [
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9
        ]

    results = pd.DataFrame({
        "actual": list(y_true),
        "probability": list(y_prob),
        "amount": list(amounts),
    })

    if results.empty:
        raise ValueError("Input data cannot be empty.")

    if results.isnull().any().any():
        raise ValueError("Input data cannot contain missing values.")

    if not results["actual"].isin([0, 1]).all():
        raise ValueError("Actual labels must be 0 or 1.")

    if not results["probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if (results["amount"] < 0).any():
        raise ValueError("Amounts cannot be negative.")

    cost_results = []

    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError(
                "Each threshold must be between 0 and 1."
            )

        # Predict fraud when probability >= threshold
        predicted = (
            results["probability"] >= threshold
        ).astype(int)

        fp_mask = (
            (results["actual"] == 0) &
            (predicted == 1)
        )

        fn_mask = (
            (results["actual"] == 1) &
            (predicted == 0)
        )

        fp_count = int(fp_mask.sum())
        fn_count = int(fn_mask.sum())

        fp_cost = fp_count * FP_COST

        fn_loss = float(
            results.loc[fn_mask, "amount"].sum()
        )

        total_cost = fp_cost + fn_loss

        cost_results.append({
            "threshold": threshold,
            "FP": fp_count,
            "FN": fn_count,
            "FP_cost_USD": fp_cost,
            "FN_loss_USD": fn_loss,
            "total_cost_USD": total_cost,
        })

    return pd.DataFrame(cost_results)

