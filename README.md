# POC1_karthick

This repository contains the notebook `POC1_karthick.ipynb` — a proof-of-concept that demonstrates a simple supervised learning workflow using student performance data.

What this notebook does
- Loads `students_performance_tracker.csv` (expects it to be in the same folder).
- Cleans missing values and drops unused name columns.
- Encodes categorical fields and prepares features and target (`gender`).
- Trains and compares three baseline classifiers: K-Nearest Neighbors, linear SVM, and Logistic Regression.
- Reports accuracy and confusion matrices for each model.

Quick instructions to run (choose one):

- Run in Google Colab (no local setup):
  1. Open `POC1_karthick.ipynb` in Colab (there is a Colab badge in the notebook).
  2. Use the upload cell to upload `students_performance_tracker.csv` when prompted.
  3. Run cells from top to bottom (Runtime ▶ Run all).

- Run locally (basic steps):
  1. Ensure Python is installed and on PATH.
  2. (Optional) Create a virtual environment and install dependencies:
	 ```bash
	 python -m venv .venv
	 source .venv/Scripts/activate   # Git Bash on Windows
	 pip install -r requirements.txt
	 ```
  3. Start Jupyter Notebook and open `POC1_karthick.ipynb`:
	 ```bash
	 jupyter notebook
	 ```
  4. Run cells from top to bottom.

Notes
- The notebook uses Latin-1 encoding when reading the CSV (see the first cells). Keep the CSV in the same folder as the notebook.
- If you prefer, run first in Colab to avoid local environment issues.

That's all this README contains — it documents `POC1_karthick` only.
