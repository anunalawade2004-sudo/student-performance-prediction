import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from data_utils import get_feature_columns, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PASS_MARK

def build_preprocessor():
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

def train_models(df):
    features = get_feature_columns(df)
    X = df[features].copy()
    y_reg = df["G3"].astype(float)
    y_cls = (df["G3"] >= PASS_MARK).astype(int)
    X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    reg = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    clf = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
    ])
    reg.fit(X_train, y_train_reg)
    clf.fit(X_train, y_train_cls)
    pred_reg = reg.predict(X_test)
    pred_cls = clf.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test_reg, pred_reg)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_reg, pred_reg))),
        "r2": float(r2_score(y_test_reg, pred_reg)),
        "accuracy": float(accuracy_score(y_test_cls, pred_cls)),
        "precision": float(precision_score(y_test_cls, pred_cls, zero_division=0)),
        "recall": float(recall_score(y_test_cls, pred_cls, zero_division=0)),
        "f1": float(f1_score(y_test_cls, pred_cls, zero_division=0)),
    }
    return reg, clf, metrics, X_test, y_test_reg, y_test_cls, pred_reg, pred_cls

def make_student_row(df, overrides):
    """Create one feature row using a representative real dataset profile, then apply user inputs."""
    features = get_feature_columns(df)
    row = {}
    for col in features:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            row[col] = float(series.median())
        else:
            row[col] = series.mode(dropna=True).iloc[0]
    row.update(overrides)
    return pd.DataFrame([row], columns=features)
