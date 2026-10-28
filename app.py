import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_dataset, FEATURE_COLUMNS, PASS_MARK
from model_utils import train_models, make_student_row, perform_clustering
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Fix metric card text visibility */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dbe3ef;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #172033 !important;
    opacity: 1 !important;
}

/* Fix success and warning message visibility */
div[data-testid="stAlert"] {
    opacity: 1 !important;
}

div[data-testid="stAlert"] p {
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Student Performance Prediction System</h1>
  <p>Predict academic grades, estimate pass/fail outcomes, compare machine-learning algorithms, and explore student data.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading student dataset...")
def get_data():
    return load_dataset()

@st.cache_resource(show_spinner="Training and evaluating machine-learning models...")
def get_results(data):
    # train_models returns a dictionary; keep all pages consistent with that format.
    return train_models(data)

@st.cache_data(show_spinner="Clustering students...")
def get_clusters(data, n_clusters):
    return perform_clustering(data, n_clusters)

try:
    data = get_data()
    results = get_results(data)
except Exception as error:
    st.error("The dataset could not be loaded or the models could not be trained.")
    st.write("If this is the first run, check your internet connection and installed packages.")
    st.exception(error)
    st.stop()

regression_models = results["regression_models"]
classification_models = results["classification_models"]
regression_metrics = results["regression_metrics"]
classification_metrics = results["classification_metrics"]
cv_results = results["cv_results"]
feature_importance = results["feature_importance"]
X_test = results["X_test"]
y_test_reg = results["y_test_reg"]
y_test_cls = results["y_test_cls"]
regression_predictions = results["regression_predictions"]
classification_predictions = results["classification_predictions"]

# Use the best-performing model according to the held-out test set.
# MAE is minimized for regression; F1-score is maximized for classification.
best_reg_name = min(regression_metrics, key=lambda name: regression_metrics[name]["MAE"])
best_cls_name = max(classification_metrics, key=lambda name: classification_metrics[name]["F1-score"])
best_reg_model = regression_models[best_reg_name]
best_cls_model = classification_models[best_cls_name]

with st.sidebar:
    st.title("🎓 ML Playground")
    st.caption("Student Academic Analytics")
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict a Student",
            "Model Comparison",
            "Cross-Validation",
            "Student Clustering",
            "Feature Importance",
            "Dataset Explorer",
            "About Project",
        ],
    )
    st.divider()
    st.markdown("### 📚 Dataset Information")
    st.write("UCI Student Performance")
    st.write(f"Records: {len(data)}")
    st.write(f"Input features: {len(FEATURE_COLUMNS)}")
    st.write(f"Pass threshold: {PASS_MARK}/20")
    st.caption("Python • Streamlit • scikit-learn")

if page == "Dashboard":
    st.subheader("📊 Project Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(data):,}")
    c2.metric("Input Features", len(FEATURE_COLUMNS))
    c3.metric("Best Regression MAE", f'{regression_metrics[best_reg_name]["MAE"]:.2f} / 20')
    c4.metric("Best Classification F1", f'{classification_metrics[best_cls_name]["F1-score"]:.3f}')
    st.caption(f"Best regression model by test MAE: {best_reg_name}. Best classification model by test F1-score: {best_cls_name}.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### 📈 Final Grade Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(data["G3"], bins=11, edgecolor="white")
        ax.set_xlabel("Final Grade (0–20)")
        ax.set_ylabel("Number of Students")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### ✅ Pass / Fail Distribution")
        counts = data["pass_fail"].map({1: "Pass", 0: "Fail"}).value_counts().reindex(["Pass", "Fail"], fill_value=0)
        st.bar_chart(pd.DataFrame({"Students": counts}))
        st.caption(f'Passed: {int(counts["Pass"])} | Below pass mark: {int(counts["Fail"])}')
    st.divider()
    st.markdown("#### 🧠 Machine-learning tasks")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="info-card"><h3>📈 Regression</h3><p>Estimates final mathematics grade (G3) on a 0–20 scale. Metrics: MAE, RMSE, and R².</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card"><h3>🎯 Classification</h3><p>Estimates whether the student reaches the defined pass mark. Metrics include accuracy, precision, recall, and F1-score.</p></div>', unsafe_allow_html=True)
    st.info("Predictions are estimates for educational demonstration and are not guaranteed outcomes.")

elif page == "Predict a Student":
    st.subheader("🔮 Predict Student Performance")
    st.write("Enter the details available below. Other input fields use typical values from the dataset.")
    with st.form("student_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Student Age", 15, 22, 17)
            studytime = st.select_slider(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {1: "Less than 2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: "More than 10 hours"}[x],
            )
            absences = st.slider("Number of Absences", 0, 30, 4)
        with col2:
            failures = st.select_slider("Previous Class Failures", options=[0, 1, 2, 3], value=0)
            goout = st.slider("Social Activity (1–5)", 1, 5, 3)
            health = st.slider("Self-Reported Health (1–5)", 1, 5, 3)
        with col3:
            internet = st.selectbox("Internet at Home", ["yes", "no"])
            higher = st.selectbox("Wants Higher Education", ["yes", "no"])
            schoolsup = st.selectbox("Extra Educational Support", ["no", "yes"])
        submitted = st.form_submit_button("✨ Predict Performance", type="primary", use_container_width=True)
    if submitted:
        student_details = {
            "age": age, "studytime": studytime, "absences": absences,
            "failures": failures, "goout": goout, "health": health,
            "internet": internet, "higher": higher, "schoolsup": schoolsup,
        }
        try:
            student_row = make_student_row(data, student_details)
            predicted_grade = float(np.clip(best_reg_model.predict(student_row)[0], 0, 20))
            predicted_class = int(best_cls_model.predict(student_row)[0])
            # Use a probability-capable model for probability display if the selected model supports it.
            probability_model = best_cls_model
            if not hasattr(probability_model, "predict_proba"):
                probability_model = classification_models["Random Forest"]
            pass_probability = float(probability_model.predict_proba(student_row)[0][1])
            st.session_state["last_student_row"] = student_row
            st.divider()
            st.subheader("📋 Prediction Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Final Grade", f"{predicted_grade:.1f} / 20")
            c2.metric("Predicted Outcome", "PASS" if predicted_class == 1 else "AT RISK")
            c3.metric("Estimated Pass Probability", f"{pass_probability * 100:.1f}%")
            st.progress(pass_probability)
            if predicted_class == 1:
                st.success("The model predicts that this student profile meets the pass threshold.")
            else:
                st.warning("The model flags this profile as potentially at risk. This is only an estimate.")
            st.caption(f"Regression model: {best_reg_name}. Classification model: {best_cls_name}. Probability is shown using {type(probability_model.named_steps['model']).__name__ if hasattr(probability_model, 'named_steps') else type(probability_model).__name__}.")
        except Exception as error:
            st.error("Prediction failed. Details are shown below to help diagnose the problem.")
            st.exception(error)

elif page == "Model Comparison":
    st.subheader("📊 Compare Machine-Learning Algorithms")
    st.markdown("### Regression")
    reg_table = pd.DataFrame(regression_metrics).T.sort_values("MAE")
    st.dataframe(reg_table.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}), use_container_width=True)
    st.caption("For MAE and RMSE, lower is better. For R², higher is better; R² can be negative.")
    fig, ax = plt.subplots(figsize=(10, 4))
    reg_table["MAE"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Regression Model Comparison (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.divider()
    st.markdown("### Classification")
    cls_table = pd.DataFrame(classification_metrics).T.sort_values("F1-score", ascending=False)
    st.dataframe(cls_table.style.format("{:.3f}"), use_container_width=True)
    st.caption("Accuracy alone may be misleading when classes are imbalanced; also consider precision, recall, and F1-score.")
    fig, ax = plt.subplots(figsize=(10, 4))
    cls_table["F1-score"].plot(kind="bar", ax=ax)
    ax.set_ylabel("F1-score")
    ax.set_title("Classification Model Comparison (higher is better)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

elif page == "Cross-Validation":
    st.subheader("🔁 Cross-Validation Results")
    st.write("Cross-validation is performed on the training split for selected classification models. The held-out test set is kept separate.")
    if cv_results:
        cv_table = pd.DataFrame(cv_results).T
        st.dataframe(cv_table.style.format({"Mean CV Accuracy": "{:.3f}", "CV Standard Deviation": "{:.3f}"}), use_container_width=True)
        st.caption("Mean CV accuracy estimates performance across folds; standard deviation indicates variation between folds.")
    else:
        st.warning("Cross-validation results are unavailable for this dataset split.")
    st.markdown("#### Held-out test classification metrics")
    st.dataframe(pd.DataFrame(classification_metrics).T.sort_values("F1-score", ascending=False).style.format("{:.3f}"), use_container_width=True)

elif page == "Student Clustering":
    st.subheader("🧩 Student Clustering (K-Means)")
    st.write("K-Means groups students by selected academic and behavioural variables. These groups are exploratory and are not pass/fail predictions.")
    n_clusters = st.slider("Number of clusters (K)", 2, 6, 3)
    try:
        clustered_data, silhouette, cluster_features = get_clusters(data, n_clusters)
        c1, c2 = st.columns(2)
        c1.metric("Number of Clusters", n_clusters)
        c2.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette is not None else "Unavailable")
        counts = clustered_data["Cluster"].value_counts().sort_index()
        st.markdown("#### Students in each cluster")
        st.bar_chart(counts.rename("Students"))
        st.markdown("#### Average profile by cluster")
        summary_cols = [c for c in ["G1", "G2", "G3", "studytime", "failures", "absences", "health"] if c in clustered_data.columns]
        st.dataframe(clustered_data.groupby("Cluster")[summary_cols].mean().round(2), use_container_width=True)
        if "G3" in clustered_data.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            for cluster_id, group in clustered_data.groupby("Cluster"):
                ax.scatter(group["G2"], group["G3"], label=f"Cluster {cluster_id}", alpha=.7)
            ax.set_xlabel("G2 (Second Period Grade)")
            ax.set_ylabel("G3 (Final Grade)")
            ax.set_title("Clusters by G2 and G3")
            ax.legend()
            ax.grid(alpha=.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.download_button("⬇️ Download Clustered Dataset", clustered_data.to_csv(index=False).encode("utf-8"), "student_clusters.csv", "text/csv")
        st.caption("Clustering uses: " + ", ".join(cluster_features))
    except Exception as error:
        st.error("Clustering could not be completed.")
        st.exception(error)

elif page == "Feature Importance":
    st.subheader("🔎 Feature Importance")
    st.write("This page shows Random Forest regression feature importances after preprocessing. Importance indicates how the fitted model used features, not causation.")
    if feature_importance is not None and not feature_importance.empty:
        top_n = st.slider("Number of features to display", 5, min(30, len(feature_importance)), min(15, len(feature_importance)))
        top_features = feature_importance.head(top_n).sort_values("Importance")
        st.dataframe(feature_importance.head(top_n).style.format({"Importance": "{:.4f}"}), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, max(4, top_n * .28)))
        ax.barh(top_features["Feature"], top_features["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances — Random Forest Regressor")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.warning("Feature importance is unavailable. Check the scikit-learn version and model preprocessing.")

elif page == "Dataset Explorer":
    st.subheader("🗂️ Dataset Explorer")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing Cells", int(data.isna().sum().sum()))
    st.markdown("### Dataset Preview")
    st.dataframe(data.head(25), use_container_width=True)
    st.markdown("### Summary Statistics")
    st.dataframe(data.describe(include="all").transpose(), use_container_width=True)
    st.markdown("### Missing Values by Column")
    missing = data.isna().sum().rename("Missing values").to_frame()
    st.dataframe(missing[missing["Missing values"] > 0] if missing["Missing values"].sum() else missing.head(0), use_container_width=True)
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Dataset CSV", csv_data, "student-mat-clean.csv", "text/csv")

elif page == "About Project":
    st.subheader("📘 About This Project")
    st.markdown("""
### Project Title
**Student Performance Prediction System Using Machine Learning**

### Objectives
- Estimate final mathematics grade using regression.
- Classify students into pass / below-pass-mark groups.
- Compare traditional, tree-based, boosting, and ensemble algorithms.
- Explore cross-validation, feature importance, and unsupervised clustering.

### Algorithms
**Regression:** Linear Regression, Ridge Regression, Random Forest, AdaBoost, Gradient Boosting, Voting Regressor.

**Classification:** Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest, AdaBoost, Gradient Boosting, Voting Classifier.

**Unsupervised learning:** K-Means clustering.

### Dataset and preprocessing
The UCI Student Performance mathematics dataset contains 395 records. Numerical values are imputed and scaled where appropriate; categorical values are imputed and one-hot encoded. G1 and G2 are excluded from supervised model inputs to reduce direct target leakage when predicting G3.

### Evaluation
Regression is evaluated with MAE, RMSE, and R². Classification is evaluated with accuracy, precision, recall, and F1-score. Cross-validation is reported for selected classification algorithms.

### Limitations
The dataset is small and comes from Portuguese schools. Results may not generalize to other populations. Model performance is not guaranteed, and predictions should not be used as the sole basis for important decisions.
""")
    st.markdown("Dataset source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)")
    st.markdown("Framework: [Streamlit documentation](https://docs.streamlit.io/)")

st.divider()
st.caption("Student Performance Prediction System | Python • Streamlit • Pandas • scikit-learn")
