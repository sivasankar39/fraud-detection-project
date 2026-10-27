# Fraud detection inference specification

- Model: `cost_sensitive_xgboost`
- Artifact: `models/fraud_xgboost_pipeline.joblib`
- Threshold: `0.050000`; probabilities >= threshold map to Fraud (1)
- Target mapping: Class 0 = Legitimate, Class 1 = Fraud

## Required input features in exact order
- `Time`
- `V1`
- `V2`
- `V3`
- `V4`
- `V5`
- `V6`
- `V7`
- `V8`
- `V9`
- `V10`
- `V11`
- `V12`
- `V13`
- `V14`
- `V15`
- `V16`
- `V17`
- `V18`
- `V19`
- `V20`
- `V21`
- `V22`
- `V23`
- `V24`
- `V25`
- `V26`
- `V27`
- `V28`
- `Amount`

## Preprocessing
The fitted pipeline owns preprocessing. Do not scale or impute separately at inference.

## Output schema
`fraud_probability`, `predicted_label`, `predicted_class`, `threshold_used`.

The illustrative threshold cost used $2 per FP and Amount as an FN-loss proxy. Confirm these assumptions with the team.
