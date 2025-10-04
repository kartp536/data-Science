# POC2_karthick — Documentation

## Purpose

This repository contains an exploratory notebook, `POC2_karthick.ipynb`, that demonstrates a supervised learning workflow to predict `gender` from student performance and demographic data.

The notebook walks through data cleaning, categorical encoding, outlier handling (IQR and z-score), several normalization/standardization experiments, skewness correction, and a comparison of model performance across experiments (KNN, linear SVM, Logistic Regression).

## Files in this folder
- `POC2_karthick.ipynb` — the main notebook with EDA, preprocessing experiments, model training and visualizations.
- `test.py` — a lightweight, runnable baseline script that reproduces the notebook's baseline training flow (KNN, SVM, Logistic Regression). It will fall back to synthetic data if the CSV is missing.
- `requirements.txt` — minimal Python dependencies needed to run the notebook or `test.py`.
- `README.md` — this document.

## Dataset

The notebook expects a CSV named `students_performance_tracker.csv` in the same folder. When using Google Colab the notebook uses the Colab file upload flow. When running locally, ensure the CSV is present (Latin-1 encoding is used in the notebook).

If you don't have the CSV, `test.py` includes a synthetic fallback so you can smoke-test the baseline pipeline.

## Quick setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows (bash): .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the baseline script (`test.py`)

The script runs a simplified pipeline matching the notebook's baseline flow:

```bash
# Run using synthetic data when CSV is not available
python test.py

# Or run using your dataset
python test.py --csv students_performance_tracker.csv
```

What `test.py` does:
- Loads the CSV if provided; otherwise generates a synthetic dataset for a smoke test.
- Drops `fNAME` and `lNAME` (if present).
- Standardizes and label-encodes categorical fields (`gender`, `residence`, `prevEducation`, `country`).
- Splits data into train/test (80/20) and trains three models: KNN, linear SVM, Logistic Regression.
- Prints accuracy scores and confusion matrices for each model.

## Notebook walkthrough (high level)

1. Imports and file upload (Colab helper is present).
2. Basic EDA: head, shape, info, missing values.
3. Cleaning: drop rows with NaN (creates `df_cleaned`), drop name columns.
4. Normalize categorical text values and label-encode relevant columns.
5. Baseline modeling: split X/y, train/test split, train KNN/SVM/LR, compute confusion matrices and accuracy, and plot results.
6. Outlier experiments:
	- IQR method: remove rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] for numeric columns and re-evaluate.
	- Z-score method: remove rows with |z| >= 2.5 for selected numeric columns and re-evaluate.
7. Normalization/standardization:
	- Plain normalize, L1, L2, and StandardScaler are applied on copies of the dataset; models are retrained and compared.
8. Skewness correction: compute skewness, apply log1p to selected columns, and visualize distributions.

## Key variables to inspect when running
- `df`, `df_cleaned`: base data states
- `df1` .. `df7`: dataset copies used for different experiments
- `x_train`, `x_test`, `y_train`, `y_test`: data splits per experiment
- Accuracy variables: `acs_knn`, `acs_svc`, `acs_lr` and their variants for each experiment (e.g., `*_out`, `*_out_z`, `*_nm`, `*_l1`, `*_l2`, `*_std`).

## Observations & caveats
- The notebook removes rows with NaN using `dropna()`. This may discard useful data — consider imputation if many rows are missing.
- `LabelEncoder` is used on `country` and other categorical columns. This assigns integer labels and can introduce unintended ordinal relationships; for nominal high-cardinality features, consider One-Hot Encoding or target encoding instead.
- Many cells mutate dataframes in-place. Restart the kernel before re-running experiments, or rely on the dataset copies (`df1..df7`) to keep experiments isolated.
- Ensure numeric columns only are used for z-score/IQR outlier detection.

## Suggested next steps / improvements
- Move preprocessing to sklearn Pipelines and ColumnTransformer to avoid in-place operations and make experiments reproducible.
- Replace LabelEncoder with OneHotEncoder (or use OrdinalEncoder carefully) where appropriate.
- Add cross-validation (GridSearchCV or cross_val_score) and hyperparameter tuning.
- Add model persistence (joblib) and an inference example cell/script.
- Replace dropna with imputation (SimpleImputer or iterative imputer) if many rows have missing values.

## Contact / Ownership
Notebook author: repository owner `kartp536` (seen in repository context).

---

If you'd like, I can also:
- Add pinned versions to `requirements.txt`.
- Convert more notebook experiments into separate runnable scripts.
- Create a small automated test (GitHub Action) that runs `test.py` as a smoke test.
