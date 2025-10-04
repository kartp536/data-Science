#!/usr/bin/env python3
"""
Small baseline runner for POC2_karthick baseline experiment.

This script attempts to load `students_performance_tracker.csv` from the current directory
(or from the path provided with --csv). If the CSV is not present, the script generates
synthetic data so the pipeline can be smoke-tested without external data.

It performs light preprocessing similar to the notebook, trains three classifiers
(KNN, linear SVM, Logistic Regression) and prints accuracy and confusion matrices.

Usage:
    python test.py [--csv PATH_TO_CSV]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        print(f"Loading CSV from: {csv_path}")
        return pd.read_csv(csv_path, encoding='latin-1')

    # fallback synthetic dataset for smoke test
    print("CSV not provided or not found. Generating synthetic dataset for smoke test.")
    n = 300
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'fNAME': [f'fn{i}' for i in range(n)],
        'lNAME': [f'ln{i}' for i in range(n)],
        'gender': rng.choice(['Male', 'Female'], size=n),
        'country': rng.integers(1, 6, size=n),
        'entryEXAM': rng.integers(40, 101, size=n),
        'studyHOURS': np.clip(rng.normal(10, 3, size=n).round().astype(int), 0, None),
        'Python': rng.integers(0, 101, size=n),
        'DB': rng.integers(0, 101, size=n),
        'Age': rng.integers(18, 40, size=n),
        'residence': rng.choice(['BI Residence', 'BI-Residence', 'Other'], size=n),
        'prevEducation': rng.choice(['High School', 'Diploma', 'Bachelors'], size=n),
    })
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NaN (same as notebook)
    df = df.dropna()

    # Drop name columns if present
    for c in ['fNAME', 'lNAME']:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Standardize gender/residence/prevEducation strings
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.lower().replace({'m': 'male', 'f': 'female'})

    if 'residence' in df.columns:
        df['residence'] = df['residence'].replace({
            'BI Residence': 'BIResidence',
            'BI-Residence': 'BIResidence',
            'BI_Residence': 'BIResidence',
        })

    if 'prevEducation' in df.columns:
        df['prevEducation'] = df['prevEducation'].replace({
            'High School': 'HighSchool',
            'DIPLOMA': 'Diploma',
            'diploma': 'Diploma',
            'Diplomaaa': 'Diploma',
            'Bachelors': 'Barrrchelors',
        })

    # Label encode categorical columns used in the notebook
    le = preprocessing.LabelEncoder()
    for col in ['gender', 'residence', 'prevEducation', 'country']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def run_baseline(df: pd.DataFrame) -> dict:
    if 'gender' not in df.columns:
        raise ValueError("The DataFrame must contain a 'gender' column as the target.")

    X = df.drop('gender', axis=1)
    y = df['gender']

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

    models = {
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'SVM': SVC(kernel='linear', C=2),
        'LogisticRegression': LogisticRegression(max_iter=1000, solver='liblinear')
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {'accuracy': float(acc), 'confusion_matrix': cm.tolist()}
        print(f"\n{name} accuracy: {acc:.4f}")
        print(f"{name} confusion matrix:\n{cm}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Run baseline experiment (KNN, SVM, LR)')
    parser.add_argument('--csv', help='path to students_performance_tracker.csv', default=None)
    args = parser.parse_args()

    try:
        df = load_data(args.csv)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)

    df = preprocess(df)

    try:
        results = run_baseline(df)
    except Exception as e:
        print(f"Error during training/evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
