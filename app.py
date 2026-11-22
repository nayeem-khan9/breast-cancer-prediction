
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =====================================================
# APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🎗️",
    layout="wide"
)


# =====================================================
# LOAD AND CACHE TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    return {
        "Logistic Regression": joblib.load(
            "models/logistic_model.pkl"
        ),

        "KNN": joblib.load(
            "models/knn_model.pkl"
        ),

        "Random Forest": joblib.load(
            "models/random_forest_model.pkl"
        )
    }


trained_models = load_models()

# Default prediction model
model = trained_models["Logistic Regression"]


# =====================================================
# LOAD AND CACHE DATASETS
# =====================================================

@st.cache_data
def load_datasets():

    full_data = pd.read_csv(
        "data/breast_cancer.csv"
    )

    test_data = pd.read_csv(
        "data/test_data.csv"
    )

    return full_data, test_data


full_data, test_data = load_datasets()

# Feature ranges for input validation
feature_data = full_data.drop(
    columns="target"
)

feature_min = feature_data.min()
feature_max = feature_data.max()


# =====================================================
# INITIALIZE SESSION STATE
# =====================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "confirm_clear_history" not in st.session_state:
    st.session_state.confirm_clear_history = False


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Prediction",
        "Model Comparison",
        "Model Evaluation",
        "Prediction History"
    ]
)


# =====================================================
# HOME PAGE - DASHBOARD
# =====================================================

if page == "Home":

    st.title("🎗️ Breast Cancer Prediction System")

    st.write(
        "A Machine Learning Based Application "
        "for Breast Cancer Prediction."
    )

    st.divider()

    # Dataset information
    total_samples = len(full_data)
    testing_samples = len(test_data)
    training_samples = total_samples - testing_samples

    # Prepare test data
    X_test = test_data.drop(
        "target",
        axis=1
    )

    y_test = test_data["target"]

    # Calculate model accuracy
    model_scores = {}

    for name, model_item in trained_models.items():

        predictions = model_item.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        ) * 100

        model_scores[name] = accuracy

    # Find best performing model
    best_model = max(
        model_scores,
        key=model_scores.get
    )

    best_accuracy = model_scores[best_model]


    # -------------------------------------------------
    # DASHBOARD STATISTICS
    # -------------------------------------------------

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(f"""
            <div style="
                background-color:#E8F1FC;
                padding:20px;
                border-radius:12px;
                border-left:5px solid #2878C8;
            ">
                <p style="color:#345678; margin:0;">
                    Total Samples
                </p>
                <h2 style="color:#174477; margin:5px 0;">
                    {total_samples}
                </h2>
                <p style="color:#567; margin:0;">
                    Dataset Records
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
            <div style="
                background-color:#E8F5EC;
                padding:20px;
                border-radius:12px;
                border-left:5px solid #299653;
            ">
                <p style="color:#345678; margin:0;">
                    Trained Models
                </p>
                <h2 style="color:#176B39; margin:5px 0;">
                    {len(trained_models)}
                </h2>
                <p style="color:#567; margin:0;">
                    ML Algorithms
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
            <div style="
                background-color:#FFF4E5;
                padding:20px;
                border-radius:12px;
                border-left:5px solid #E89B28;
            ">
                <p style="color:#76501C; margin:0;">
                    Training Samples
                </p>
                <h2 style="color:#A66308; margin:5px 0;">
                    {training_samples}
                </h2>
                <p style="color:#76501C; margin:0;">
                    Training Dataset
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col4:

        st.markdown(f"""
            <div style="
                background-color:#F1EAFE;
                padding:20px;
                border-radius:12px;
                border-left:5px solid #8655C7;
            ">
                <p style="color:#59417C; margin:0;">
                    Testing Samples
                </p>
                <h2 style="color:#6236A0; margin:5px 0;">
                    {testing_samples}
                </h2>
                <p style="color:#59417C; margin:0;">
                    Testing Dataset
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()


    # -------------------------------------------------
    # MODEL OVERVIEW
    # -------------------------------------------------

    st.subheader("🤖 Model Overview")

    model_col1, model_col2, model_col3 = st.columns(3)

    model_cards = [
        (
            "Logistic Regression",
            "Linear Classification Algorithm",
            "#2878C8"
        ),
        (
            "K-Nearest Neighbors",
            "Distance-Based Classification",
            "#299653"
        ),
        (
            "Random Forest",
            "Ensemble Learning Algorithm",
            "#8655C7"
        )
    ]

    for col, (name, description, color) in zip(
        [model_col1, model_col2, model_col3],
        model_cards
    ):

        with col:

            st.markdown(f"""
                <div style="
                    background-color:#FFFFFF;
                    padding:22px;
                    border-radius:12px;
                    border:1px solid #E0E5EB;
                    border-top:5px solid {color};
                    min-height:135px;
                ">
                    <h4 style="color:#263445; margin-top:0;">
                        {name}
                    </h4>
                    <p style="color:#667085; margin-bottom:0;">
                        {description}
                    </p>
                </div>
            """, unsafe_allow_html=True)

    st.divider()


    # -------------------------------------------------
    # BEST PERFORMING MODEL
    # -------------------------------------------------

    st.subheader("🏆 Best Performing Model")

    st.markdown(f"""
        <div style="
            background:linear-gradient(120deg,#E8F5EC,#E8F1FC);
            padding:25px;
            border-radius:14px;
            border:1px solid #D5E5DC;
            text-align:center;
        ">
            <p style="color:#39734D; font-size:15px;">
                TOP PERFORMING ALGORITHM
            </p>
            <h2 style="color:#174477; margin:8px 0;">
                {best_model}
            </h2>
            <h1 style="color:#21864A; margin:8px 0;">
                {best_accuracy:.2f}%
            </h1>
            <p style="color:#567; margin-bottom:0;">
                Model Accuracy on Test Dataset
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()


    # -------------------------------------------------
    # PREDICTION ACTIVITY
    # -------------------------------------------------

    st.subheader("📋 Prediction Activity")

    history = st.session_state.prediction_history

    total_predictions = len(history)

    malignant_count = sum(
        1 for record in history
        if record["Prediction"] == "Malignant"
    )

    benign_count = sum(
        1 for record in history
        if record["Prediction"] == "Benign"
    )

    activity_col1, activity_col2, activity_col3 = st.columns(3)

    activity_col1.metric(
        "Total Predictions",
        total_predictions
    )

    activity_col2.metric(
        "Malignant Results",
        malignant_count
    )

    activity_col3.metric(
        "Benign Results",
        benign_count
    )

    st.divider()


    # -------------------------------------------------
    # MODEL ACCURACY COMPARISON
    # -------------------------------------------------

    st.subheader("📈 Model Accuracy Comparison")

    accuracy_df = pd.DataFrame({
        "Model": list(model_scores.keys()),
        "Accuracy (%)": list(model_scores.values())
    })

    chart_col, table_col = st.columns([2, 1])

    with chart_col:

        st.bar_chart(
            accuracy_df.set_index("Model"),
            y_label="Accuracy (%)",
            height=350
        )

    with table_col:

        st.markdown("#### Accuracy Details")

        for name, score in model_scores.items():

            st.write(f"**{name}**")

            st.progress(
                min(score / 100, 1.0)
            )

            st.caption(
                f"{score:.2f}%"
            )

    st.divider()


    # -------------------------------------------------
    # EDUCATIONAL DISCLAIMER
    # -------------------------------------------------

    st.info(
        "This application is developed for educational purposes only. "
        "It is not a medical diagnostic tool."
    )


    # -------------------------------------------------
    # DASHBOARD FOOTER
    # -------------------------------------------------

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:15px; color:#7A869A;">
            <p style="margin:0;">
                Breast Cancer Prediction System
            </p>
            <p style="margin:5px 0 0; font-size:13px;">
                Developed using Python, Streamlit and Machine Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("🔬 Breast Cancer Prediction")

    st.write(
        "Enter tumor measurements to predict the tumor classification."
    )

    st.subheader("Tumor Measurements")


    # -------------------------------------------------
    # FEATURE NAMES
    # -------------------------------------------------

    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter',
        'mean area', 'mean smoothness', 'mean compactness',
        'mean concavity', 'mean concave points', 'mean symmetry',
        'mean fractal dimension', 'radius error', 'texture error',
        'perimeter error', 'area error', 'smoothness error',
        'compactness error', 'concavity error', 'concave points error',
        'symmetry error', 'fractal dimension error', 'worst radius',
        'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry',
        'worst fractal dimension'
    ]


    # -------------------------------------------------
    # SAMPLE VALUES
    # -------------------------------------------------

    sample_values = [
        17.99, 10.38, 122.8, 1001.0, 0.1184,
        0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622,
        0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]


    # -------------------------------------------------
    # PREDICTION FORM
    # -------------------------------------------------

    with st.form("prediction_form"):

        input_values = []

        cols = st.columns(3)

        for i, feature in enumerate(feature_names):

            with cols[i % 3]:

                value = st.number_input(
                    feature.title(),
                    min_value=float(feature_min[feature]),
                    max_value=float(feature_max[feature]),
                    value=float(sample_values[i]),
                    format="%.6f"
                )

                input_values.append(value)

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True
        )


    # -------------------------------------------------
    # PROCESS PREDICTION
    # -------------------------------------------------

    if submitted:

        input_data = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        prediction = model.predict(input_data)[0]

        prediction_proba = model.predict_proba(input_data)[0]

        confidence = max(prediction_proba) * 100


        # Save prediction to history
        record = {
            "Date & Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "Prediction": (
                "Malignant" if prediction == 0 else "Benign"
            ),

            "Confidence (%)": round(confidence, 2),

            **dict(zip(feature_names, input_values))
        }

        st.session_state.prediction_history.append(record)


        # -------------------------------------------------
        # DISPLAY PREDICTION RESULT
        # -------------------------------------------------

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 0:

                st.error(
                    "🔴 Malignant (Cancerous)"
                )

            else:

                st.success(
                    "🟢 Benign (Non-Cancerous)"
                )

        with col2:

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        st.warning(
            "This result is for educational purposes only, "
            "not a medical diagnosis."
        )


# =====================================================
# MODEL COMPARISON PAGE
# =====================================================

elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    st.write(
        "Compare the performance of the three trained Machine Learning models."
    )

    st.divider()


    # -------------------------------------------------
    # PREPARE TEST DATA
    # -------------------------------------------------

    X_test = test_data.drop(
        "target",
        axis=1
    )

    y_test = test_data["target"]


    # -------------------------------------------------
    # GENERATE PREDICTIONS
    # -------------------------------------------------

    logistic_pred = trained_models[
        "Logistic Regression"
    ].predict(X_test)

    knn_pred = trained_models[
        "KNN"
    ].predict(X_test)

    random_forest_pred = trained_models[
        "Random Forest"
    ].predict(X_test)


    # -------------------------------------------------
    # CALCULATE MODEL PERFORMANCE
    # -------------------------------------------------

    comparison_df = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "KNN",
            "Random Forest"
        ],

        "Accuracy": [
            accuracy_score(y_test, logistic_pred) * 100,
            accuracy_score(y_test, knn_pred) * 100,
            accuracy_score(y_test, random_forest_pred) * 100
        ],

        "Precision": [
            precision_score(
                y_test, logistic_pred,
                average="weighted", zero_division=0
            ) * 100,

            precision_score(
                y_test, knn_pred,
                average="weighted", zero_division=0
            ) * 100,

            precision_score(
                y_test, random_forest_pred,
                average="weighted", zero_division=0
            ) * 100
        ],

        "Recall": [
            recall_score(
                y_test, logistic_pred,
                average="weighted", zero_division=0
            ) * 100,

            recall_score(
                y_test, knn_pred,
                average="weighted", zero_division=0
            ) * 100,

            recall_score(
                y_test, random_forest_pred,
                average="weighted", zero_division=0
            ) * 100
        ],

        "F1-Score": [
            f1_score(
                y_test, logistic_pred,
                average="weighted", zero_division=0
            ) * 100,

            f1_score(
                y_test, knn_pred,
                average="weighted", zero_division=0
            ) * 100,

            f1_score(
                y_test, random_forest_pred,
                average="weighted", zero_division=0
            ) * 100
        ]

    })


    # -------------------------------------------------
    # PERFORMANCE COMPARISON CHART
    # -------------------------------------------------

    st.subheader("📈 Performance Comparison Chart")

    chart_df = comparison_df.set_index("Model")

    st.bar_chart(
        chart_df,
        y_label="Performance (%)",
        height=400
    )

    st.divider()


    # -------------------------------------------------
    # PERFORMANCE TABLE
    # -------------------------------------------------

    st.subheader("📋 Model Performance Table")

    st.dataframe(
        comparison_df.style
        .format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%"
        })
        .background_gradient(
            cmap="Blues",
            subset=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score"
            ]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "All performance metrics are calculated using the test dataset. "
        "Precision, Recall and F1-Score use weighted averaging "
        "to consider both classes."
    )


# =====================================================
# MODEL EVALUATION PAGE
# =====================================================

elif page == "Model Evaluation":

    st.title("📈 Model Evaluation")

    st.write(
        "Evaluate the performance of the three trained Machine Learning models."
    )

    st.divider()


    # -------------------------------------------------
    # PREPARE TEST DATA
    # -------------------------------------------------

    X_test = test_data.drop(
        "target",
        axis=1
    )

    y_test = test_data["target"]


    # -------------------------------------------------
    # GENERATE PREDICTIONS
    # -------------------------------------------------

    logistic_pred = trained_models[
        "Logistic Regression"
    ].predict(X_test)

    knn_pred = trained_models[
        "KNN"
    ].predict(X_test)

    random_forest_pred = trained_models[
        "Random Forest"
    ].predict(X_test)


    # -------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------

    st.subheader("🔍 Select Model")

    model_options = [
        "Logistic Regression",
        "KNN",
        "Random Forest"
    ]

    selected_model = st.radio(
        "Choose a model to evaluate:",
        model_options,
        horizontal=True
    )


    # Select predictions
    predictions_by_model = {
        "Logistic Regression": logistic_pred,
        "KNN": knn_pred,
        "Random Forest": random_forest_pred
    }

    predictions = predictions_by_model[selected_model]


    # -------------------------------------------------
    # SELECTED MODEL INFORMATION
    # -------------------------------------------------

    st.success(
        f"Currently Evaluating: **{selected_model}**"
    )

    st.divider()


    # -------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------

    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Malignant",
            "Actual Benign"
        ],
        columns=[
            "Predicted Malignant",
            "Predicted Benign"
        ]
    )


    # Style confusion matrix
    def style_confusion_matrix(row):

        if row.name == "Actual Malignant":

            color = "#F0F6FC"
            text_color = "#172B4D"

        else:

            color = "#174477"
            text_color = "#FFFFFF"

        return [
            f"background-color: {color}; color: {text_color}"
        ] * len(row)


    st.dataframe(
        cm_df.style
        .format("{:.0f}")
        .apply(
            style_confusion_matrix,
            axis=1
        ),
        use_container_width=True
    )

    st.caption(
        "The matrix shows the actual and predicted classifications "
        "of Malignant and Benign cases."
    )

    st.divider()


    # -------------------------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------------------------

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "Malignant",
            "Benign"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()


    # Calculate average scores for color mapping
    score_columns = [
        "precision",
        "recall",
        "f1-score"
    ]

    row_scores = report_df[
        score_columns
    ].mean(axis=1)

    min_score = row_scores.min()
    max_score = row_scores.max()


    # Style classification report
    def style_classification_report(row):

        score = row_scores.loc[row.name]

        colors = [
            "#F0F6FC",
            "#D9E8F5",
            "#B8D4EB",
            "#8DB9DC",
            "#5D99C8"
        ]

        if pd.isna(score) or max_score == min_score:

            color_index = 2

        else:

            color_index = round(
                (score - min_score) /
                (max_score - min_score) * 4
            )

        color = colors[color_index]

        return [
            f"background-color: {color}; color: #172B4D"
        ] * len(row)


    st.dataframe(
        report_df.style
        .format("{:.4f}")
        .apply(
            style_classification_report,
            axis=1
        ),
        use_container_width=True
    )

    st.caption(
        "Precision, Recall and F1-score measure the classification "
        "performance of each class. Support represents the number "
        "of actual samples."
    )


# =====================================================
# PREDICTION HISTORY PAGE
# =====================================================

elif page == "Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "View and manage your previous prediction records."
    )

    st.divider()

    history = st.session_state.prediction_history


    # -------------------------------------------------
    # DISPLAY HISTORY
    # -------------------------------------------------

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True
        )

        col1, col2 = st.columns([3, 1])

        with col1:

            st.success(
                f"Total Predictions: {len(history)}"
            )

        with col2:

            if st.button(
                "🗑️ Clear History",
                use_container_width=True
            ):

                st.session_state.confirm_clear_history = True


        # -------------------------------------------------
        # CLEAR HISTORY CONFIRMATION
        # -------------------------------------------------

        if st.session_state.confirm_clear_history:

            st.warning(
                "Are you sure you want to delete all prediction records?"
            )

            confirm_col, cancel_col = st.columns(2)

            with confirm_col:

                if st.button(
                    "Yes, Clear History",
                    type="primary"
                ):

                    st.session_state.prediction_history = []

                    st.session_state.confirm_clear_history = False

                    st.rerun()

            with cancel_col:

                if st.button("Cancel"):

                    st.session_state.confirm_clear_history = False

                    st.rerun()


    # -------------------------------------------------
    # EMPTY HISTORY
    # -------------------------------------------------

    else:

        st.info(
            "No prediction history available yet."
        )
