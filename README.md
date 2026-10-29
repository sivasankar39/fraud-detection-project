# FraudGuard AI — Credit Card Fraud Detection & Monitoring

FraudGuard AI is a web-based application for exploring credit card fraud detection, analyzing model predictions, evaluating business costs, and monitoring transaction data. It combines data analysis and machine learning with an interactive dashboard to help users understand fraud patterns and the trade-offs involved in fraud detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [Application Modules](#application-modules)
- [Business Cost Assumptions](#business-cost-assumptions)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

## Project Overview

Credit card fraud detection is challenging because fraudulent transactions represent only a small fraction of all transactions. A model must identify suspicious activity while minimizing false alarms and missed fraud cases.

FraudGuard AI provides an interactive interface for examining fraud patterns, running model-based predictions, exploring precision-recall trade-offs, estimating business costs, and reviewing transaction records.

The project uses a trained XGBoost-based machine learning pipeline for transaction predictions and provides supporting data analysis and monitoring functionality.

## Key Features

- **Dashboard:** Overview of dataset statistics and fraud detection indicators.
- **Fraud Analytics:** Explore transaction data and analyze fraudulent transactions.
- **Cost Optimizer:** Examine how classification thresholds and business cost assumptions affect false positives, false negatives, and estimated total cost.
- **Precision–Recall Explorer:** Explore the relationship between precision and recall at different decision thresholds.
- **Live Simulator:** Replay transactions from the available dataset and display model predictions as the simulation progresses.
- **Investigation Queue:** Review and filter fraud investigation records, including transaction amounts and priority classifications.
- **Drift Monitor:** Examine changes between reference and comparison data windows, with feature-level drift analysis being integrated.

## Technology Stack

### Programming and application development

- Python
- Flask
- HTML and CSS
- JavaScript

### Data analysis and machine learning

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SciPy, where required by the analysis modules

### Visualization and monitoring

- Plotly
- Matplotlib
- Seaborn
- Evidently AI, for applicable monitoring workflows
- SHAP, for applicable model interpretation workflows

### Development tools

- Git and GitHub
- Visual Studio Code or another Python-compatible IDE
- Python virtual environment

Install the dependencies specified by the repository's `requirements.txt` file. Some libraries above may be used only by notebooks or supporting analysis scripts rather than the Flask application itself.

## Dataset

The project is based on a credit card transaction dataset containing:

- **284,807** transaction records in the original dataset.
- **492** fraudulent transactions in the original dataset.
- **31** columns, including `Time`, `V1`–`V28`, `Amount`, and `Class`.

The `Class` column indicates the transaction label: `0` represents a legitimate transaction and `1` represents a fraudulent transaction.

The original dataset has a highly imbalanced class distribution, making precision, recall, and precision-recall analysis particularly relevant.

The project also uses prepared data splits and saved model artifacts for inference and evaluation.

> **Important:** Use the actual data and model files supplied with the repository. If these files are stored using Git LFS or are not included in the repository, follow the corresponding setup instructions below.

## Project Structure

The following is a high-level guide to the main project components. Exact file locations may vary as the project evolves.

fraud-detection-project/
├── FraudGuardAI/
│ ├── app.py
│ └── templates/
│ ├── Dashboard.html
│ ├── Fraud_Analytics.html
│ ├── cost_optimizer.html
│ ├── PR_Explorer.html
│ ├── Live_Simulator.html
│ ├── Investigation_Queue.html
│ └── drift_monitor.html
├── data/
│ └── Dataset and prepared data files
├── models/
│ ├── fraud_xgboost_pipeline.joblib
│ └── fraud_xgboost_metadata.json
├── src/
│ └── Data processing, monitoring and analysis modules
├── results/
│ └── Model predictions and evaluation results
├── reports/
│ └── Model inference documentation
├── docs/
│ └── Project documentation
├── requirements.txt
└── README.md

This is an illustrative structure, not a guarantee that every listed folder or filename is present in every branch.

## Prerequisites

Install the following before setting up the project:

1. **Python:** Use a version compatible with the project's dependencies and saved model artifacts.
2. **Git:** Required to clone the repository.
3. **Git LFS:** Required if the repository stores large datasets or model files using Git Large File Storage.
4. **Terminal:** PowerShell on Windows, or a terminal on Linux/macOS.

Check your Python and Git installations:

python --version
git --version

## Installation and Setup

### 1. Clone the repository

Replace the placeholder with the actual GitHub repository URL.

git clone <REPOSITORY_URL>
cd fraud-detection-project

If you already have the repository locally, open a terminal in its root directory.

### 2. Download Git LFS files, if applicable

If the repository uses Git LFS, install Git LFS first and then run:

git lfs install
git lfs pull

Verify that large data files and model artifacts are actual files rather than Git LFS pointer files.

### 3. Create a virtual environment

On Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

On Linux or macOS:

python3 -m venv .venv
source .venv/bin/activate

If PowerShell prevents virtual environment activation, use the Python executable inside `.venv` directly or follow your system's approved script-execution policy.

### 4. Install dependencies

With the virtual environment activated, upgrade pip:

python -m pip install --upgrade pip

Install the project's dependencies:

python -m pip install -r requirements.txt

If the repository does not contain `requirements.txt`, the project maintainers should provide a tested dependency list before a fresh installation can be considered reproducible. Avoid installing arbitrary package versions without checking compatibility with the saved model.

### 5. Verify the required model files

The application uses the saved model pipeline and associated metadata.

Check that the repository contains the required files, including:

models/fraud_xgboost_pipeline.joblib
models/fraud_xgboost_metadata.json

Also check `reports/inference_spec.md` for details about the model's expected input features and prediction requirements.

If a model file is missing, obtain the correct artifact from the project maintainers or the repository's documented storage location. Do not replace it with an unrelated model.

### 6. Verify the dataset files

Check that the required dataset and prepared input files are available in the locations expected by the application.

Depending on the module, the project may use the original transaction dataset, prepared training/validation/test files, or saved prediction results.

If a file is missing, consult the project's data-loading code and documentation to determine its expected path. Do not rename or move files arbitrarily, as that may break the application.

## Running the Application

From the repository root, activate the virtual environment and start the Flask application:

python FraudGuardAI/app.py

If startup succeeds, Flask will display the local address in the terminal. Open that address in your web browser; the default development address is commonly:

http://127.0.0.1:5000/

Keep the terminal running while using the application. Stop the server with `Ctrl+C`.

Run the application using Flask's development server for local testing only. A production deployment requires an appropriate production server and deployment configuration.

## Application Modules

### Dashboard

Provides an overview of the dataset and key fraud-related indicators.

### Fraud Analytics

Allows users to explore transaction records and examine fraudulent transaction patterns.

### Cost Optimizer

Calculates estimated business costs under configurable assumptions and decision thresholds. It helps demonstrate that the threshold giving the highest recall or precision does not necessarily minimize estimated business cost.

### Precision–Recall Explorer

Displays precision-recall information to help users compare model behavior at different thresholds, particularly when fraud cases are rare.

### Live Simulator

Replays transactions from the available dataset and displays model-based predictions as the simulation progresses. It is a dataset replay demonstration, not a connection to a live payment processing system.

### Investigation Queue

Provides an interface for reviewing and filtering investigation records and their assigned priorities. The availability of review-status updates may depend on the implementation.

### Drift Monitor

Provides a monitoring interface for comparing transaction data across reference and comparison windows. Feature-level metrics, such as Population Stability Index (PSI) and Kolmogorov–Smirnov (KS) statistics, may be used to assess distribution changes as the integration is completed.

## Business Cost Assumptions

The cost analysis uses configurable assumptions to illustrate the financial consequences of classification errors.

The documented baseline assumptions are:

- **False positive:** An assumed cost of USD 2 per legitimate transaction incorrectly flagged as fraudulent.
- **False negative:** The transaction amount is used as the estimated loss for a fraudulent transaction that is missed.

These are illustrative project assumptions, not verified real-world banking costs. Results depend on the selected threshold, the transaction data, model predictions, and configured cost values.

## Troubleshooting

**1. `ModuleNotFoundError`**

Confirm that the virtual environment is activated and that dependencies were installed using the project's requirements file.

**2. Model file not found**

Check that the required `.joblib` model and metadata files exist at the expected paths. Download missing Git LFS artifacts if applicable.

**3. Dataset file not found**

Verify the data paths used by the application and ensure the required dataset files are available.

**4. Application starts but a page fails**

Check the terminal for the traceback, confirm the relevant template exists, and verify that the page's required data or model artifact is available.

**5. Package compatibility errors**

Check the project's supported Python version and dependency versions. Recreate the virtual environment with a compatible Python version if necessary.

**6. Port 5000 is already in use**

Stop the other process using the port or configure a different port in the application's development settings.

## Disclaimer

FraudGuard AI is an educational and demonstration project. Model predictions, thresholds, cost estimates, and monitoring results should not be treated as production banking decisions or financial advice. Real-world deployment would require additional validation, security controls, privacy safeguards, monitoring, and operational testing.
