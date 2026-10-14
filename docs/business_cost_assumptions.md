# Business Cost Assumptions for Fraud Detection

## 1. Purpose

The purpose of this document is to define the initial business cost assumptions for the Credit Card Fraud Detection project. These assumptions will be used to estimate the financial impact of false positives and false negatives and to support cost-based model evaluation.

## 2. False Positive (FP) Cost

**Definition:** A false positive occurs when a genuine transaction is incorrectly classified as fraudulent.

**Business impact:** The business may need to spend time and resources investigating a transaction that is actually genuine.

**Initial assumption:** The cost of investigating one false positive is assumed to be **$2 per transaction**.

**Justification:** A fixed provisional cost is used to represent the operational effort involved in reviewing a false alarm.

**Status:** This is an illustrative project assumption, not a verified industry figure or an actual bank expense. It may be revised if reliable business data becomes available.

## 3. False Negative (FN) Cost

**Definition:** A false negative occurs when a fraudulent transaction is incorrectly classified as genuine.

**Business impact:** The fraudulent transaction may go undetected, resulting in potential financial loss.

**Initial estimation method:** The full transaction amount of each missed fraudulent transaction will be used as a proxy for potential fraud loss.

**Formula:**

Estimated FN Cost = Sum of the transaction amounts of all false negatives.

**Justification:** The transaction amount is available in the dataset and provides a simple starting point for estimating potential loss.

**Status:** This is a simplified estimate, not a measurement of actual financial loss. The actual loss may be lower if the transaction is recovered or refunded.

## 4. Currency and Units

According to the Kaggle dataset description, the `Amount` column represents transaction amounts in **US dollars (USD)**.

The false-positive cost is provisionally assumed to be $2 per transaction. The false-negative cost is estimated using the full transaction amount of each missed fraudulent transaction.

Both cost components are therefore expressed in USD and can be combined to calculate the estimated total cost.

## 6. Total Cost Evaluation

The total estimated cost is calculated as:

**Total Estimated Cost = Total FP Cost + Total FN Cost**

Where:

- **Total FP Cost** = Number of False Positives × Assumed FP Cost per Transaction
- **Total FN Cost** = Sum of the transaction amounts of all False Negatives

The calculations can be repeated at different classification thresholds to compare the estimated business cost of different model decisions.

The resulting total is expressed in USD. It is an estimate based on the project's provisional cost assumptions, not a measurement of actual bank losses or operating expenses.

## 7. Limitations

- The FP cost of $2 per transaction is an illustrative assumption and has not been validated using actual business data.
- The FN cost uses transaction amounts as a proxy for potential fraud loss.
- Actual fraud losses, recoveries, and refunds are unavailable.
- The estimated total cost does not include other expenses, such as customer inconvenience, dispute handling, or fraud recovery.
- The assumptions may be updated after team review or when reliable business information becomes available.

## 8. Review Status

These assumptions are provisional and intended for the initial project analysis. They should be reviewed by the project team before being treated as finalized business cost parameters.
