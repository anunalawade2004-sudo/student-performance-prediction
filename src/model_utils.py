
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score

from .data_utils import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    get_feature_columns,
)


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def make_pipeline(model, selector=None):
    steps = [("preprocessor", build_preprocessor())]

    if selector is not None:
        steps.append(("selector", selector))

    steps.append(("model", model))
    return Pipeline(steps)


def train_models(df):
    # Predict numerical final marks (G3), not pass/fail.
    X = df[get_feature_columns(df)].copy()
    y = df["G3"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    regression_models = {
        "Multiple Linear Regression": make_pipeline(
            LinearRegression()
        ),
        "Multiple Linear Regression + SelectKBest": make_pipeline(
            LinearRegression(),
            SelectKBest(score_func=f_regression, k=10),
        ),
        "AdaBoost Regression": make_pipeline(
            AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.5,
                random_state=42,
            )
        ),
    }

    regression_metrics = {}
    regression_predictions = {}

    for name, model in regression_models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        regression_predictions[name] = predictions
        regression_metrics[name] = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2 Score": float(r2_score(y_test, predictions)),
        }

    # Five-fold cross-validation using R².
    # Preprocessing and feature selection happen inside each fold's pipeline.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in regression_models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )

        cv_results[name] = {
            "Fold 1 R2": float(scores[0]),
            "Fold 2 R2": float(scores[1]),
            "Fold 3 R2": float(scores[2]),
            "Fold 4 R2": float(scores[3]),
            "Fold 5 R2": float(scores[4]),
            "Mean R2": float(np.mean(scores)),
            "Std R2": float(np.std(scores, ddof=1)),
        }

    # Get feature scores from the fitted SelectKBest pipeline.
    selector_model = regression_models[
        "Multiple Linear Regression + SelectKBest"
    ]
    preprocessor = selector_model.named_steps["preprocessor"]
    selector = selector_model.named_steps["selector"]

    feature_names = preprocessor.get_feature_names_out()
    feature_scores = pd.DataFrame({
        "Feature": feature_names,
        "Score": selector.scores_,
    })

    feature_scores = (
        feature_scores
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )

    # Choose the best model by held-out test MAE for the prediction demo.
    best_regression_name = min(
        regression_metrics,
        key=lambda name: regression_metrics[name]["MAE"],
    )

    return {
        "regression_models": regression_models,
        "regression_metrics": regression_metrics,
        "regression_predictions": regression_predictions,
        "X_test": X_test,
        "y_test_reg": y_test,
        "cv_results": cv_results,
        "feature_importance": feature_scores,
        "features": list(X.columns),
        "best_regression_name": best_regression_name,
    }


def make_student_row(df, overrides=None):
    overrides = overrides or {}
    row = {}

    for column in NUMERIC_FEATURES:
        row[column] = float(df[column].median())

    for column in CATEGORICAL_FEATURES:
        modes = df[column].mode(dropna=True)
        row[column] = modes.iloc[0] if not modes.empty else "unknown"

    row.update(overrides)
    return pd.DataFrame(
        [row],
        columns=get_feature_columns(df),
    )


def perform_clustering(df, n_clusters=3):
    cluster_features = [
        column
        for column in [
            "G1", "G2", "G3", "studytime",
            "failures", "absences", "health",
        ]
        if column in df.columns
    ]

    cluster_data = df[cluster_features].copy()
    cluster_data = SimpleImputer(
        strategy="median"
    ).fit_transform(cluster_data)
    cluster_data = StandardScaler().fit_transform(cluster_data)

    actual_clusters = min(n_clusters, len(df) - 1)
    actual_clusters = max(2, actual_clusters)

    clusterer = KMeans(
        n_clusters=actual_clusters,
        random_state=42,
        n_init=10,
    )
    labels = clusterer.fit_predict(cluster_data)

    result = df.copy()
    result["Cluster"] = labels

    if 1 < len(set(labels)) < len(labels):
        score = float(silhouette_score(cluster_data, labels))
    else:
        score = float("nan")

    return result, score, cluster_features
