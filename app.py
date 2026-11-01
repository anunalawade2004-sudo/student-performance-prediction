
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import (
    load_dataset,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from src.model_utils import (
    train_models,
    make_student_row,
    perform_clustering,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Prediction System")
st.caption(
    "Machine Learning project using Multiple Linear Regression, "
    "AdaBoost Regression, SelectKBest and K-Means Clustering."
)


# --------------------------------------------------
# LOAD DATA AND TRAIN MODELS
# --------------------------------------------------

@st.cache_data
def get_data():
    return load_dataset()


@st.cache_resource
def get_results(data):
    return train_models(data)


@st.cache_data
def get_clusters(data, number):
    return perform_clustering(data, number)


try:
    df = get_data()
    results = get_results(df)
except Exception as error:
    st.error(f"Unable to load the dataset or train the models: {error}")
    st.info(
        "Check that your data folder contains student-mat.csv "
        "and that all required Python packages are installed."
    )
    st.stop()


regression_models = results["regression_models"]
regression_metrics = results["regression_metrics"]
regression_predictions = results["regression_predictions"]
y_test = results["y_test_reg"]
cv_results = results["cv_results"]
feature_importance = results["feature_importance"]
best_name = results["best_regression_name"]


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📚 Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Predict Student Marks",
        "Model Comparison",
        "Cross-Validation",
        "SelectKBest",
        "Student Clustering",
        "Dataset Explorer",
        "About Project",
    ],
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", len(df))

    with col2:
        st.metric("Input Features", len(results["features"]))

    with col3:
        st.metric("Regression Models", len(regression_models))

    with col4:
        st.metric("Target Variable", "G3")

    st.divider()

    st.subheader("📌 Project Overview")

    st.write(
        """
        This system predicts a student's final academic grade using
        machine learning regression algorithms.

        **Main objectives:**
        - Predict final marks (G3) on a scale of 0 to 20.
        - Compare different regression models.
        - Evaluate models using MAE, RMSE and R² Score.
        - Validate model performance using 5-fold cross-validation.
        - Identify useful input features using SelectKBest.
        - Discover student performance groups using K-Means clustering.
        """
    )

    st.subheader("🏆 Best Model by Test MAE")

    st.success(f"Best model in this comparison: {best_name}")

    best_metrics = regression_metrics[best_name]

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{best_metrics['MAE']:.3f}")
    col2.metric("RMSE", f"{best_metrics['RMSE']:.3f}")
    col3.metric("R² Score", f"{best_metrics['R2 Score']:.3f}")

    st.caption(
        "MAE is used to select the displayed best model in this demo. "
        "A lower MAE indicates smaller average prediction errors. "
        "R² measures how well the model explains variation in final marks."
    )

    st.subheader("📈 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# PREDICT STUDENT MARKS
# --------------------------------------------------

elif page == "Predict Student Marks":

    st.header("🔮 Predict Student Final Marks")

    st.write(
        "Enter student details to estimate the final grade (G3)."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=15,
                max_value=22,
                value=17,
            )

            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                index=1,
                format_func=lambda x: {
                    1: "Less than 2 hours",
                    2: "2 to 5 hours",
                    3: "5 to 10 hours",
                    4: "More than 10 hours",
                }[x],
            )

            failures = st.number_input(
                "Past Class Failures",
                min_value=0,
                max_value=4,
                value=0,
            )

            absences = st.number_input(
                "Number of Absences",
                min_value=0,
                max_value=100,
                value=0,
            )

            goout = st.slider(
                "Going Out With Friends (1–5)",
                min_value=1,
                max_value=5,
                value=3,
            )

            health = st.slider(
                "Current Health (1–5)",
                min_value=1,
                max_value=5,
                value=3,
            )

        with col2:
            school = st.selectbox(
                "School",
                sorted(df["school"].dropna().unique().tolist()),
            )

            sex = st.selectbox(
                "Sex",
                sorted(df["sex"].dropna().unique().tolist()),
            )

            address = st.selectbox(
                "Home Address Type",
                sorted(df["address"].dropna().unique().tolist()),
            )

            internet = st.selectbox(
                "Internet Access at Home",
                sorted(df["internet"].dropna().unique().tolist()),
            )

            higher = st.selectbox(
                "Wants Higher Education",
                sorted(df["higher"].dropna().unique().tolist()),
            )

            schoolsup = st.selectbox(
                "Extra Educational Support",
                sorted(df["schoolsup"].dropna().unique().tolist()),
            )

        selected_model_name = st.selectbox(
            "Choose Regression Model",
            list(regression_models.keys()),
        )

        submitted = st.form_submit_button(
            "Predict Final Marks",
            use_container_width=True,
        )

    if submitted:

        overrides = {
            "age": age,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "goout": goout,
            "health": health,
            "school": school,
            "sex": sex,
            "address": address,
            "internet": internet,
            "higher": higher,
            "schoolsup": schoolsup,
        }

        try:
            student_row = make_student_row(df, overrides)

            model = regression_models[selected_model_name]
            prediction = float(model.predict(student_row)[0])

            # Keep the displayed prediction within the grade scale.
            prediction = float(np.clip(prediction, 0, 20))

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Final Grade (G3)",
                    f"{prediction:.2f} / 20",
                )

            with col2:
                st.metric(
                    "Selected Model",
                    selected_model_name,
                )

            st.progress(prediction / 20)

            if prediction >= 10:
                st.success(
                    "The predicted grade is at or above 10/20."
                )
            else:
                st.warning(
                    "The predicted grade is below 10/20."
                )

            st.caption(
                "This is a machine learning estimate, not a guaranteed "
                "exam result. The model uses the supplied details and "
                "typical values for unspecified features."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("📊 Regression Model Comparison")

    st.write(
        "Compare model performance using MAE, RMSE and R² Score."
    )

    metrics_df = pd.DataFrame(regression_metrics).T
    metrics_df.index.name = "Regression Model"

    st.subheader("Evaluation Metrics")
    st.dataframe(
        metrics_df.round(4),
        use_container_width=True,
    )

    st.caption(
        "Lower MAE and RMSE are better. A higher R² Score generally "
        "indicates a better fit. R² is not classification accuracy."
    )

    st.divider()

    # Metric comparison graph
    metric = st.selectbox(
        "Choose metric to visualize",
        ["MAE", "RMSE", "R2 Score"],
    )

    metric_values = {
        name: values[metric]
        for name, values in regression_metrics.items()
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        list(metric_values.keys()),
        list(metric_values.values()),
    )

    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Actual versus predicted graph
    st.subheader("🎨 Actual vs Predicted Marks")

    selected_model = st.selectbox(
        "View actual versus predicted marks",
        list(regression_models.keys()),
    )

    actual = np.asarray(y_test)
    predicted = np.asarray(
        regression_predictions[selected_model]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    points = ax.scatter(
        actual,
        predicted,
        c=actual,
        cmap="plasma",
        s=75,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )

    ax.plot(
        [0, 20],
        [0, 20],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction",
    )

    fig.colorbar(
        points,
        ax=ax,
        label="Actual Final Grade",
    )

    ax.set_title(
        f"Actual vs Predicted Marks — {selected_model}"
    )
    ax.set_xlabel("Actual Final Grade")
    ax.set_ylabel("Predicted Final Grade")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"**Best model by test MAE:** {best_name}")

    st.caption(
        "The best model is selected using test-set MAE for this demo. "
        "For formal model selection, use validation data or "
        "cross-validation before evaluating on a final test set."
    )


# --------------------------------------------------
# CROSS-VALIDATION
# --------------------------------------------------

elif page == "Cross-Validation":

    st.header("🔁 5-Fold Cross-Validation")

    st.write(
        """
        Cross-validation divides the dataset into five parts.
        Each model is trained on four parts and evaluated on the
        remaining part. This process repeats five times.
        """
    )

    cv_df = pd.DataFrame(cv_results).T
    cv_df.index.name = "Regression Model"

    st.subheader("Cross-Validation R² Scores")

    st.dataframe(
        cv_df.round(4),
        use_container_width=True,
    )

    st.caption(
        "Mean R² is the average across the five folds. "
        "The standard deviation indicates how much the scores vary."
    )

    st.subheader("Mean R² Comparison")

    mean_scores = {
        name: values["Mean R2"]
        for name, values in cv_results.items()
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        list(mean_scores.keys()),
        list(mean_scores.values()),
    )

    ax.set_title("5-Fold Cross-Validation — Mean R²")
    ax.set_ylabel("Mean R² Score")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# SELECTKBEST FEATURE SELECTION
# --------------------------------------------------

elif page == "SelectKBest":

    st.header("🎯 Feature Selection Using SelectKBest")

    st.write(
        """
        SelectKBest ranks input features using the F-regression score.
        The regression pipeline selects the 10 highest-scoring
        transformed features before fitting Linear Regression.
        """
    )

    if feature_importance.empty:
        st.warning("No feature scores are available.")
    else:
        st.subheader("Feature Scores")

        number_of_features = min(
            15,
            len(feature_importance),
        )

        top_features = feature_importance.head(
            number_of_features
        ).sort_values("Score", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            top_features["Feature"],
            top_features["Score"],
        )

        ax.set_title("Top Features Selected by F-Regression Score")
        ax.set_xlabel("F-Regression Score")
        ax.set_ylabel("Feature")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature Score Table")

        st.dataframe(
            feature_importance.round(4),
            use_container_width=True,
        )

    st.info(
        "The F-regression score measures the strength of the "
        "feature's individual linear relationship with G3. "
        "It is not model accuracy and does not prove causation."
    )


# --------------------------------------------------
# STUDENT CLUSTERING
# --------------------------------------------------

elif page == "Student Clustering":

    st.header("👥 Student Performance Clustering")

    st.write(
        """
        K-Means groups students based on similarities in their
        academic grades and selected student-related attributes.
        Clustering is an unsupervised learning technique.
        """
    )

    number_of_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=6,
        value=3,
    )

    try:
        clustered_df, silhouette, cluster_features = get_clusters(
            df,
            number_of_clusters,
        )

        st.subheader("Clustering Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Number of Clusters",
                clustered_df["Cluster"].nunique(),
            )

        with col2:
            if np.isfinite(silhouette):
                st.metric(
                    "Silhouette Score",
                    f"{silhouette:.3f}",
                )
            else:
                st.metric(
                    "Silhouette Score",
                    "Not available",
                )

        st.caption(
            "A silhouette score closer to 1 generally indicates "
            "better-separated clusters. It is not classification accuracy."
        )

        st.subheader("Students in Each Cluster")

        cluster_counts = (
            clustered_df["Cluster"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(cluster_counts)

        st.subheader("Average Marks by Cluster")

        average_marks = clustered_df.groupby("Cluster")[
            ["G1", "G2", "G3"]
        ].mean()

        st.dataframe(
            average_marks.round(2),
            use_container_width=True,
        )

        st.subheader("Cluster Visualization")

        x_axis = st.selectbox(
            "Choose horizontal axis",
            cluster_features,
            index=0,
        )

        y_axis = st.selectbox(
            "Choose vertical axis",
            cluster_features,
            index=min(1, len(cluster_features) - 1),
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        points = ax.scatter(
            clustered_df[x_axis],
            clustered_df[y_axis],
            c=clustered_df["Cluster"],
            cmap="viridis",
            s=60,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.3,
        )

        fig.colorbar(
            points,
            ax=ax,
            label="Cluster",
        )

        ax.set_title("Student Clusters")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Clustered Dataset")

        st.dataframe(
            clustered_df,
            use_container_width=True,
        )

    except Exception as error:
        st.error(f"Clustering failed: {error}")


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Dataset Explorer":

    st.header("📁 Dataset Explorer")

    st.write(
        "Explore the student dataset used by the regression models."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True,
    )

    st.subheader("Final Grade Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        df["G3"],
        bins=11,
        edgecolor="black",
    )

    ax.set_title("Distribution of Final Grades (G3)")
    ax.set_xlabel("Final Grade")
    ax.set_ylabel("Number of Students")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Between Grades")

    grade_columns = [
        column
        for column in ["G1", "G2", "G3"]
        if column in df.columns
    ]

    correlation = df[grade_columns].corr()

    st.dataframe(
        correlation.round(3),
        use_container_width=True,
    )


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.header("ℹ️ About the Project")

    st.subheader("Student Performance Prediction System")

    st.write(
        """
        This project applies machine learning techniques to estimate
        students' final academic marks using demographic, family,
        social and school-related information.
        """
    )

    st.subheader("Algorithms and Techniques")

    st.markdown(
        """
        **1. Multiple Linear Regression**
        - Predicts final marks using input features.

        **2. AdaBoost Regression**
        - Combines sequentially trained weak regressors to improve
          prediction performance.

        **3. SelectKBest Feature Selection**
        - Ranks transformed features using F-regression scores.
        - Used with Multiple Linear Regression.

        **4. K-Means Clustering**
        - Groups students based on similarities in selected attributes.
        """
    )

    st.subheader("Evaluation Metrics")

    st.markdown(
        """
        - **MAE:** Mean Absolute Error; lower is better.
        - **RMSE:** Root Mean Squared Error; lower is better.
        - **R² Score:** Measures how much variation in final marks
          the regression model explains.
        - **5-Fold Cross-Validation:** Evaluates performance across
          five different train/validation splits.
        - **Silhouette Score:** Measures cluster separation.
        """
    )

    st.subheader("Dataset Information")

    st.write(
        """
        The project uses the UCI Student Performance dataset.
        The target variable is G3, the final-period grade on a
        0–20 scale. G1 and G2 are excluded from regression inputs
        to avoid using earlier grades as direct predictors in this
        particular model setup.
        """
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - Python
        - Streamlit
        - Pandas and NumPy
        - Matplotlib
        - Scikit-learn
        """
    )

    st.caption(
        "This project is designed for academic learning and "
        "demonstration. Predictions should not be treated as "
        "definitive judgments about a student's ability."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Student Performance Prediction System")
st.sidebar.caption("Regression-Based Machine Learning Project")
