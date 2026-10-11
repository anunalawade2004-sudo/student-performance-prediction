
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_utils import load_dataset, FEATURE_COLUMNS, PASS_MARK
from model_utils import train_models, make_student_row

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #2563eb);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 25px;
        color: white;
    }

    .hero h1 {
        color: white !important;
        font-size: 36px;
    }

    .hero p {
        color: #e0e7ff !important;
        font-size: 17px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #111827 !important;
    }

    div[data-testid="stMetric"] label p {
        color: #475569 !important;
    }

    h1, h2, h3 {
        color: #172554;
    }

    div[data-testid="stDataFrame"] {
        background: white;
        border-radius: 12px;
    }

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>
        Predict academic grades, estimate pass/fail outcomes,
        and explore student data using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA AND TRAIN MODELS
# --------------------------------------------------

@st.cache_data(show_spinner="Loading student dataset...")
def get_data():
    return load_dataset()


@st.cache_resource(show_spinner="Training machine learning models...")
def get_models(data):
    return train_models(data)


try:
    data = get_data()

    (
        regression_model,
        classification_model,
        metrics,
        X_test,
        y_test_reg,
        y_test_cls,
        y_pred_reg,
        y_pred_cls,
    ) = get_models(data)

except Exception as error:
    st.error("Unable to load the dataset or train the models.")
    st.write("Check your internet connection if this is the first run.")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.title("🎓 ML Playground")
    st.caption("Student Academic Analytics")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict a Student",
            "Model Evaluation",
            "Dataset Explorer",
            "About Project",
        ],
    )

    st.divider()

    st.markdown("### 📚 Dataset Information")
    st.write("UCI Student Performance")
    st.write(f"Total records: {len(data)}")
    st.write(f"Input features: {len(FEATURE_COLUMNS)}")
    st.write(f"Pass threshold: {PASS_MARK}/20")

    st.divider()

    st.caption("Python | Streamlit | scikit-learn")

# --------------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("📊 Project Dashboard")

    st.write(
        "An overview of the student dataset and "
        "the performance of the trained models."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Students in Dataset",
        f"{len(data):,}",
    )

    col2.metric(
        "Input Features",
        f"{len(FEATURE_COLUMNS)}",
    )

    col3.metric(
        "Regression MAE",
        f"{metrics['mae']:.2f} / 20",
    )

    col4.metric(
        "Classification Accuracy",
        f"{metrics['accuracy'] * 100:.1f}%",
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        st.subheader("📈 Final Grade Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.hist(
            data["G3"],
            bins=11,
            edgecolor="white",
        )

        ax.set_xlabel("Final Grade (0–20)")
        ax.set_ylabel("Number of Students")
        ax.set_title("Distribution of Final Grades")
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:

        st.subheader("✅ Pass / Fail Distribution")

        counts = (
            data["pass_fail"]
            .map({1: "Pass", 0: "Fail"})
            .value_counts()
            .reindex(["Pass", "Fail"], fill_value=0)
        )

        chart_data = pd.DataFrame({
            "Outcome": counts.index,
            "Students": counts.values,
        }).set_index("Outcome")

        st.bar_chart(
            chart_data,
            y="Students",
            use_container_width=True,
        )

        st.caption(
            f"Passed: {int(counts['Pass'])} students | "
            f"Failed: {int(counts['Fail'])} students"
        )

    st.divider()

    st.subheader("🧠 Machine Learning Tasks")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Regression</h3>
            <p>
                Estimates a student's final mathematics grade
                on a scale from 0 to 20.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Classification</h3>
            <p>
                Predicts whether a student is likely to pass
                or fall below the defined passing threshold.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Predictions are estimates for educational demonstration "
        "and should not be treated as guaranteed outcomes."
    )

# --------------------------------------------------
# PAGE 2: PREDICT A STUDENT
# --------------------------------------------------

elif page == "Predict a Student":

    st.subheader("🔮 Predict Student Performance")

    st.write(
        "Enter the student's details to estimate their final grade "
        "and pass/fail outcome."
    )

    with st.form("student_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.slider(
                "Student Age",
                15, 22, 17,
            )

            studytime = st.select_slider(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    1: "Less than 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "More than 10 hours",
                }[x],
            )

            absences = st.slider(
                "Number of Absences",
                0, 30, 4,
            )

        with col2:

            failures = st.select_slider(
                "Previous Class Failures",
                options=[0, 1, 2, 3],
                value=0,
            )

            goout = st.slider(
                "Social Activity (1–5)",
                1, 5, 3,
            )

            health = st.slider(
                "Self-Reported Health (1–5)",
                1, 5, 3,
            )

        with col3:

            internet = st.selectbox(
                "Internet at Home",
                ["yes", "no"],
            )

            higher = st.selectbox(
                "Wants Higher Education",
                ["yes", "no"],
            )

            schoolsup = st.selectbox(
                "Extra Educational Support",
                ["no", "yes"],
            )

        submitted = st.form_submit_button(
            "✨ Predict Performance",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        student_details = {
            "age": age,
            "studytime": studytime,
            "absences": absences,
            "failures": failures,
            "goout": goout,
            "health": health,
            "internet": internet,
            "higher": higher,
            "schoolsup": schoolsup,
        }

        try:

            student_row = make_student_row(
                data,
                student_details,
            )

            predicted_grade = float(
                np.clip(
                    regression_model.predict(student_row)[0],
                    0,
                    20,
                )
            )

            predicted_class = int(
                classification_model.predict(student_row)[0]
            )

            pass_probability = float(
                classification_model.predict_proba(
                    student_row
                )[0][1]
            )

            st.divider()
            st.subheader("📋 Prediction Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Predicted Final Grade",
                f"{predicted_grade:.1f} / 20",
            )

            c2.metric(
                "Predicted Outcome",
                "PASS" if predicted_class == 1 else "AT RISK",
            )

            c3.metric(
                "Estimated Pass Probability",
                f"{pass_probability * 100:.1f}%",
            )

            st.progress(pass_probability)

            if predicted_class == 1:
                st.success(
                    "The model predicts a passing outcome "
                    "for this student profile."
                )
            else:
                st.warning(
                    "The model flags this profile as potentially "
                    "at risk. This is only an estimate."
                )

            st.caption(
                "The dataset's original grade scale is 0–20. "
                "Features not entered in this form use typical "
                "dataset values through the project's helper function."
            )

        except Exception as error:
            st.error("Prediction failed. Please check the input features.")
            st.exception(error)

# --------------------------------------------------
# PAGE 3: MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.subheader("📉 Model Evaluation")

    st.write(
        "These results measure performance on a held-out test set."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"{metrics['mae']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.3f}")
    c3.metric("R² Score", f"{metrics['r2']:.3f}")
    c4.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.1f}%",
    )

    st.divider()

    st.subheader("Actual vs Predicted Grades")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.scatter(
        y_test_reg,
        y_pred_reg,
        alpha=0.75,
    )

    ax.plot(
        [0, 20],
        [0, 20],
        linestyle="--",
    )

    ax.set_xlabel("Actual Grade")
    ax.set_ylabel("Predicted Grade")
    ax.set_title("Regression Model Performance")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test_cls,
        y_pred_cls,
        labels=[0, 1],
        target_names=["Fail / At Risk", "Pass"],
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True,
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test_cls,
        y_pred_cls,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fail / At Risk", "Pass"],
    ).plot(
        ax=ax,
        colorbar=False,
    )

    ax.set_title("Classification Confusion Matrix")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Precision",
        f"{precision_score(y_test_cls, y_pred_cls, zero_division=0):.3f}",
    )

    p2.metric(
        "Recall",
        f"{recall_score(y_test_cls, y_pred_cls, zero_division=0):.3f}",
    )

    p3.metric(
        "F1 Score",
        f"{f1_score(y_test_cls, y_pred_cls, zero_division=0):.3f}",
    )

    st.caption(
        "Metrics can vary by train/test split and do not guarantee "
        "performance on students from other schools."
    )

# --------------------------------------------------
# PAGE 4: DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.subheader("🗂️ Dataset Explorer")

    st.write(
        "Explore the UCI Student Performance dataset used "
        "to train the prediction models."
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
    )

    st.markdown("### Dataset Shape")

    c1, c2 = st.columns(2)

    c1.metric("Rows", data.shape[0])
    c2.metric("Columns", data.shape[1])

    st.markdown("### Summary Statistics")

    st.dataframe(
        data.describe(include="all").transpose(),
        use_container_width=True,
    )

    csv_data = data.drop(
        columns=["pass_fail"],
        errors="ignore",
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset CSV",
        data=csv_data,
        file_name="student-mat-clean.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# PAGE 5: ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.subheader("📘 About This Project")

    st.markdown("""
### Project Title
**Student Performance Prediction System Using Machine Learning**

### Objective
To build an interactive application that estimates students' final
mathematics grades and classifies their outcomes as pass or fail.

### Algorithms Used

**1. Random Forest Regression**
- Predicts the final grade, G3, on a 0–20 scale.
- Evaluated using MAE, RMSE, and R².

**2. Random Forest Classification**
- Predicts whether a student meets the passing threshold.
- Evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

### Dataset
The UCI Student Performance dataset for mathematics contains 395 records.

### Preprocessing
- Numerical features are imputed when necessary.
- Categorical features are encoded by the model pipeline.
- Earlier and final grade columns are excluded from input features to reduce target leakage.

### Limitations
The dataset is relatively small and comes from Portuguese schools.
Results may not generalize to other student populations. Predictions are
estimates and should not be used as the sole basis for important decisions.
""")

    st.markdown(
        "[UCI Student Performance Dataset]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

    st.markdown(
        "[Streamlit Documentation](https://docs.streamlit.io/)"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Python • Streamlit • Pandas • scikit-learn"
)
