# Student Performance Prediction System

An interactive machine learning web application that predicts student final mathematics grades and pass/fail outcomes using supervised learning.

## Features

- Predict final grade (0–20) with Random Forest Regression.
- Predict pass/fail with Random Forest Classification.
- Interactive Streamlit form.
- Model evaluation dashboard with regression metrics, classification report and confusion matrix.
- Dataset explorer and CSV download.
- Loads the official UCI dataset on first run and caches it locally.

## Dataset

**UCI Student Performance** — Mathematics subset (`student-mat.csv`), containing 395 students and 33 original attributes.

- Official source: https://archive.ics.uci.edu/dataset/320/student+performance
- Original final-grade target: `G3`, from 0 to 20.
- Classification rule: Pass if `G3 >= 10`; Fail / at risk if `G3 < 10`.
- The app excludes `G1`, `G2` and `G3` from input features to reduce target leakage and make early prediction more meaningful.

The dataset is downloaded to `data/student-mat.csv` when the app first runs. An internet connection is required for the first run unless you manually place the dataset file there.

## Requirements

- Python 3.10 or newer
- VS Code (recommended)
- Internet access for first dataset download

## Run locally on Windows PowerShell

```powershell
# 1. Create and enter the project folder in VS Code terminal
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, use Command Prompt instead:
# .venv\Scripts\activate.bat

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Start the web app
python -m streamlit run app.py
```

The app should open in your browser. Streamlit's official instructions: https://docs.streamlit.io/

## Evaluate models from terminal

```powershell
python train_model.py
```

This prints MAE, RMSE, R², accuracy, precision, recall and F1-score for the held-out test split.

## Project structure

```text
Student-Performance-Prediction/
├── app.py
├── data_utils.py
├── model_utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── (student-mat.csv downloads on first run)
└── models/
```

## GitHub upload

Create an empty repository on GitHub, for example `student-performance-prediction`. Then run these commands from the project folder:

```powershell
git init
git add .
git commit -m "Build student performance prediction app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to authenticate, follow the browser sign-in prompt.

## Limitations and responsible use

The data comes from two Portuguese secondary schools and is relatively small. Results may not generalize to other schools or countries. The compact prediction form fills fields not entered by the user with typical values from the dataset, so outputs are estimates. Do not use the model to punish, label, or make high-stakes decisions about students.

## Tools

Python · Pandas · scikit-learn · Streamlit · Matplotlib
