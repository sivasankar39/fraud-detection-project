
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

    # FN loss remains in dataset amount units until the currency
    # and compatibility of units have been verified.
    total_fn_loss = results.loc[fn_mask, "amount"].sum()

    print("False Positives:", fp_count)
    print("False Negatives:", fn_count)
    print(f"Estimated FP cost: ${total_fp_cost:.2f} (provisional)")
    print(f"Estimated FN loss: {total_fn_loss:.2f} (dataset amount units)")
    print("Note: Do not combine these costs until their units are compatible.")

    return {
        "fp_count": fp_count,
        "fn_count": fn_count,
        "total_fp_cost": total_fp_cost,
        "total_fn_loss": total_fn_loss
    }
