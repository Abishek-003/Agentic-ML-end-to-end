import traceback
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils.data_loader import load_dataset
from agents.dataset_analyzer import generate_metadata, analyze_dataset
from agents.feature_engineer import apply_feature_engineering
from agents.feature_cleaner_agent import analyze_features_with_llm
from utils.feature_cleaner import analyze_features, drop_selected_columns
from utils.preprocessing import preprocess_dataset
from utils.imbalance_handler import (
    check_class_imbalance,
    should_apply_smote,
    get_class_weight_flag,
)
from agents.feature_selector import (
    mutual_information_selection,
    rfe_selection,
    feature_importance_ranking,
)
from utils.dimensionality_reduction import apply_pca, apply_ica, get_pca_variance
from agents.model_trainer import train_models


def build_analysis_cache_key(df, target_column, description):
    cache_version = "v4_problem_type_fix"
    dataset_hash = str(
        pd.util.hash_pandas_object(df, index=True).sum()
    )
    return (
        f"{cache_version}|"
        f"{dataset_hash}|"
        f"{target_column}|"
        f"{description.strip()}"
    )


def infer_problem_type_heuristic(y):
    y_non_null = y.dropna()
    total_count = int(len(y_non_null))
    if total_count == 0:
        return "classification", "heuristic_empty_default"

    unique_count = int(y_non_null.nunique())

    if (
        pd.api.types.is_object_dtype(y_non_null)
        or pd.api.types.is_categorical_dtype(y_non_null)
        or pd.api.types.is_bool_dtype(y_non_null)
    ):
        return "classification", "heuristic_categorical"

    if pd.api.types.is_integer_dtype(y_non_null):
        if unique_count <= min(20, max(2, int(total_count * 0.05))):
            return "classification", "heuristic_integer_discrete"
        return "regression", "heuristic_integer_continuous"

    if pd.api.types.is_float_dtype(y_non_null):
        low_cardinality_cutoff = min(
            20,
            max(
                2,
                int(total_count * 0.05)
            )
        )

        is_integer_like = np.all(
            np.isclose(
                y_non_null.values,
                np.round(y_non_null.values),
                atol=1e-9,
            )
        )

        if unique_count <= low_cardinality_cutoff:
            return "classification", "heuristic_float_integer_like"

        if not is_integer_like:
            return "regression", "heuristic_float_continuous"

        return "regression", "heuristic_float_continuous"

    return "regression", "heuristic_fallback"


def canonicalize_analysis(analysis):
    if not isinstance(analysis, dict):
        return {}

    result = dict(analysis)
    features = result.get("feature_engineering", [])
    if isinstance(features, list):
        clean_features = [
            item for item in features
            if isinstance(item, dict)
        ]
        result["feature_engineering"] = sorted(
            clean_features,
            key=lambda item: json.dumps(
                item,
                sort_keys=True
            )
        )
    return result


def resolve_problem_type(
    y,
    analysis_problem_type,
    user_problem_type_choice,
):
    if user_problem_type_choice == "Classification":
        return "classification", "user"

    if user_problem_type_choice == "Regression":
        return "regression", "user"

    heuristic_type, heuristic_reason = infer_problem_type_heuristic(y)
    llm_type = (
        analysis_problem_type
        if analysis_problem_type in ["classification", "regression"]
        else None
    )

    if llm_type is None:
        return heuristic_type, heuristic_reason

    if heuristic_type != llm_type:
        return heuristic_type, f"{heuristic_reason}_override_llm"

    return llm_type, "llm"


def reset_pipeline_state():
    preserve_keys = {
        "uploaded_file",
        "description",
    }

    for key in list(
        st.session_state.keys()
    ):
        if key not in preserve_keys:
            del st.session_state[key]


def load_tabular_file(file_obj):

    filename = (
        file_obj.name
        .lower()
    )

    if filename.endswith(".csv"):
        return pd.read_csv(file_obj)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file_obj)

    raise ValueError(
        "Unsupported file format. Upload CSV or Excel."
    )


def build_model_bundle(
        *,
        best_model_name,
        best_model,
        preprocessor,
        label_encoder,
        target_column,
        problem_type,
        analysis,
        raw_feature_columns,
        preprocessor_feature_columns,
        selected_feature_columns,
        final_feature_columns,
        feature_selection_method,
        dimensionality_method,
        dimensionality_model,
        apply_feature_engineering_enabled,
        columns_to_drop,
        description,
        cleaned_df
):

    return {

        "best_model_name":
        best_model_name,

        "model":
        best_model,

        "preprocessor":
        preprocessor,

        "label_encoder":
        label_encoder,

        "target_column":
        target_column,

        "problem_type":
        problem_type,

        "analysis":
        analysis,

        "raw_feature_columns":
        raw_feature_columns,

        "preprocessor_feature_columns":
        preprocessor_feature_columns,

        "selected_feature_columns":
        selected_feature_columns,

        "final_feature_columns":
        final_feature_columns,

        "feature_selection_method":
        feature_selection_method,

        "dimensionality_method":
        dimensionality_method,

        "dimensionality_model":
        dimensionality_model,

        "apply_feature_engineering_enabled":
        apply_feature_engineering_enabled,

        "columns_to_drop":
        columns_to_drop,

        "description":
        description,

        "training_signature":
        build_analysis_cache_key(
            cleaned_df,
            target_column,
            description
        )
    }


def prepare_prediction_frame(
        input_df,
        bundle
):

    working_df = (
        input_df.copy()
    )

    target_column = bundle.get(
        "target_column"
    )

    raw_feature_columns = bundle.get(
        "raw_feature_columns",
        []
    )

    if target_column in working_df.columns:
        working_df = working_df.drop(
            columns=[
                target_column
            ],
            errors="ignore"
        )

    working_df = working_df.reindex(
        columns=raw_feature_columns
    )

    if bundle.get(
        "apply_feature_engineering_enabled",
        False
    ):
        working_df = apply_feature_engineering(
            working_df,
            bundle.get(
                "analysis",
                {}
            )
        )

    preprocessor = bundle.get(
        "preprocessor"
    )

    preprocessor_feature_columns = bundle.get(
        "preprocessor_feature_columns",
        []
    )

    transformed = preprocessor.transform(
        working_df
    )

    transformed_df = pd.DataFrame(
        transformed,
        columns=preprocessor_feature_columns,
        index=working_df.index
    )

    selected_feature_columns = bundle.get(
        "selected_feature_columns",
        preprocessor_feature_columns
    )

    transformed_df = transformed_df.reindex(
        columns=selected_feature_columns,
        fill_value=0
    )

    dimensionality_method = bundle.get(
        "dimensionality_method",
        "None"
    )
    dimensionality_model = bundle.get(
        "dimensionality_model"
    )

    if (
        dimensionality_method == "PCA"
        or dimensionality_method == "ICA"
    ) and dimensionality_model is not None:

        reduced = (
            dimensionality_model.transform(
                transformed_df
            )
        )

        final_feature_columns = bundle.get(
            "final_feature_columns"
        )

        if not final_feature_columns:
            prefix = (
                "PCA"
                if dimensionality_method == "PCA"
                else "ICA"
            )
            final_feature_columns = [
                f"{prefix}_{index + 1}"
                for index in range(
                    reduced.shape[1]
                )
            ]

        transformed_df = pd.DataFrame(
            reduced,
            columns=final_feature_columns,
            index=working_df.index
        )

    return transformed_df


def generate_predictions(
        input_df,
        bundle
):

    prediction_input = (
        prepare_prediction_frame(
            input_df,
            bundle
        )
    )

    model = bundle.get(
        "model"
    )

    predictions = (
        model.predict(
            prediction_input
        )
    )

    if (
        bundle.get(
            "problem_type"
        )
        == "classification"
        and bundle.get(
            "label_encoder"
        ) is not None
    ):
        predictions = (
            bundle[
                "label_encoder"
            ]
            .inverse_transform(
                pd.Series(
                    predictions
                )
                .astype(int)
            )
        )

    result_df = (
        input_df.copy()
    )

    result_df[
        "prediction"
    ] = predictions

    if (
        bundle.get(
            "problem_type"
        )
        == "classification"
        and hasattr(
            model,
            "predict_proba"
        )
    ):
        try:
            probabilities = (
                model.predict_proba(
                    prediction_input
                )
            )

            if len(
                probabilities.shape
            ) == 2:
                result_df[
                    "prediction_confidence"
                ] = (
                    np.max(
                        probabilities,
                        axis=1
                    )
                )
        except Exception:
            pass

    target_column = bundle.get(
        "target_column"
    )

    if target_column in result_df.columns:
        actual_col = (
            f"actual_{target_column}"
        )
        result_df.rename(
            columns={
                target_column:
                actual_col
            },
            inplace=True
        )

        if (
            bundle.get(
                "problem_type"
            )
            == "regression"
            and actual_col in result_df.columns
        ):
            result_df[
                "residual"
            ] = (
                result_df[
                    actual_col
                ]
                -
                result_df[
                    "prediction"
                ]
            )
        elif actual_col in result_df.columns:
            result_df[
                "correct"
            ] = (
                result_df[
                    actual_col
                ].astype(str)
                ==
                result_df[
                    "prediction"
                ].astype(str)
            )

    return result_df


def render_single_prediction_inputs(
        feature_columns,
        reference_df
):

    input_values = {}

    feature_layout = st.columns(2)

    for index, column_name in enumerate(feature_columns):

        reference_series = (
            reference_df[column_name]
            if column_name in reference_df.columns
            else pd.Series(dtype="object")
        )

        non_null_values = (
            reference_series.dropna()
        )

        with feature_layout[index % 2]:

            if pd.api.types.is_bool_dtype(reference_series):
                input_values[column_name] = st.selectbox(
                    f"{column_name}",
                    [False, True],
                    key=f"single_pred_{column_name}"
                )

            elif pd.api.types.is_numeric_dtype(reference_series):

                if pd.api.types.is_integer_dtype(reference_series):
                    default_value = (
                        int(
                            non_null_values.median()
                        )
                        if len(non_null_values) > 0
                        else 0
                    )
                    input_values[column_name] = st.number_input(
                        f"{column_name}",
                        value=default_value,
                        step=1,
                        key=f"single_pred_{column_name}"
                    )
                else:
                    default_value = (
                        float(
                            non_null_values.median()
                        )
                        if len(non_null_values) > 0
                        else 0.0
                    )
                    input_values[column_name] = st.number_input(
                        f"{column_name}",
                        value=default_value,
                        key=f"single_pred_{column_name}"
                    )

            else:
                options = (
                    non_null_values.astype(str)
                    .unique()
                    .tolist()
                )

                if 0 < len(options) <= 20:
                    input_values[column_name] = st.selectbox(
                        f"{column_name}",
                        options,
                        key=f"single_pred_{column_name}"
                    )
                else:
                    default_value = (
                        str(
                            non_null_values.mode().iloc[0]
                        )
                        if len(non_null_values) > 0
                        else ""
                    )
                    input_values[column_name] = st.text_input(
                        f"{column_name}",
                        value=default_value,
                        key=f"single_pred_{column_name}"
                    )

    return input_values


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Agentic ML Pipeline", layout="wide")

st.title("Agentic ML Pipeline")
st.markdown(
    """
Upload dataset -> Analyze dataset -> Optional Feature Engineering ->
Feature Selection -> PCA / ICA -> Model Training -> Leaderboard
"""
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Pipeline Configuration")

problem_type_choice = st.sidebar.selectbox(
    "Problem Type",
    ["Auto (LLM + Heuristic)", "Classification", "Regression"],
    key="problem_type_choice",
)

feature_selection_method = st.sidebar.selectbox(
    "Feature Selection",
    ["None", "Mutual Information", "RFE"],
    key="feature_selection_method",
)

num_features = st.sidebar.number_input(
    "Features To Keep",
    min_value=1,
    max_value=1000,
    value=10,
    key="num_features",
)

dimensionality_method = st.sidebar.selectbox(
    "Dimensionality Reduction",
    ["None", "PCA", "ICA"],
    key="dimensionality_method",
)

n_components = st.sidebar.slider(
    "Number of Components",
    min_value=2,
    max_value=50,
    value=5,
    key="n_components",
)

imbalance_strategy = st.sidebar.selectbox(
    "Imbalance Handling",
    ["Auto", "None", "Class Weights", "SMOTE"],
    key="imbalance_strategy",
)

run_llm_cleaner = st.sidebar.checkbox(
    "Run LLM Feature Cleaner",
    value=False,
    key="run_llm_cleaner",
)

apply_feature_engineering_enabled = st.sidebar.checkbox(
    "Apply Feature Engineering",
    value=True,
    key="apply_feature_engineering_enabled",
)

st.sidebar.button(
    "Reset Pipeline",
    use_container_width=True,
    on_click=reset_pipeline_state,
)


# =====================================================
# DATASET INPUT
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"],
    key="uploaded_file",
)

description = st.text_area(
    "Dataset Description",
    placeholder="Describe the ML problem...",
    key="description",
)


# =====================================================
# MAIN PIPELINE
# =====================================================

if uploaded_file:
    try:
        df = load_dataset(uploaded_file)

        st.success("Dataset Loaded Successfully")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # ==========================================
        # TARGET SELECTION
        # ==========================================

        st.subheader("Target Variable Selection")
        st.markdown("**Select the label/target column carefully.**")
        target_column = st.selectbox(
            "**Select Target Column**",
            df.columns.tolist(),
            key="target_column",
        )

        # ==========================================
        # FEATURE CLEANER
        # ==========================================

        st.subheader("Feature Cleaner Suggestions")
        feature_report = analyze_features(df, target_column)
        st.json(feature_report["analysis"])

        st.markdown("**Columns To Drop**")
        columns_to_drop = st.multiselect(
            "**Columns To Drop**",
            feature_report["suggested_drop_columns"],
            default=[],
            key="columns_to_drop",
        )
        st.caption(
            "Only clearly unusable columns are suggested for dropping. "
            "Features are not dropped automatically."
        )

        review_columns = feature_report.get("review_columns", [])
        if review_columns:
            st.info(
                "Review-only columns (kept by default): "
                + ", ".join(review_columns)
            )

        if target_column in columns_to_drop:
            st.warning("Target column cannot be dropped.")
            columns_to_drop = [
                col for col in columns_to_drop
                if col != target_column
            ]

        cleaned_df = drop_selected_columns(df, columns_to_drop)
        st.success(f"Remaining Features: {cleaned_df.shape[1]}")

        # ==========================================
        # OPTIONAL LLM CLEANER
        # ==========================================

        if run_llm_cleaner:
            with st.spinner("Running LLM Feature Cleaner..."):
                llm_feature_report = analyze_features_with_llm(
                    cleaned_df,
                    target_column,
                    description,
                )
                st.subheader("LLM Feature Cleaner Output")
                st.json(llm_feature_report)

        # ==========================================
        # DATASET ANALYSIS
        # ==========================================

        if st.button("Analyze Dataset", key="analyze_dataset"):
            analysis_status = st.empty()
            analysis_status.info(
                "Analyzing dataset... reading the selected target column and metadata."
            )

            with st.spinner(
                "Analyzing dataset... this may take a moment."
            ):
                metadata = generate_metadata(cleaned_df)
                cache_key = build_analysis_cache_key(
                    cleaned_df,
                    target_column,
                    description
                )
                analysis_cache = st.session_state.setdefault("analysis_cache", {})

                if cache_key in analysis_cache:
                    analysis = analysis_cache[cache_key]
                    st.info(
                        "Using cached analysis for this same dataset, target, and description."
                    )
                else:
                    analysis = analyze_dataset(
                        metadata,
                        description,
                        target_column
                    )
                    analysis = canonicalize_analysis(analysis)
                    analysis_cache[cache_key] = analysis

            analysis_status.empty()
            st.session_state["analysis"] = analysis
            st.success("Dataset Analysis Complete")
            st.json(analysis)

        # =====================================================
        # PIPELINE EXECUTION
        # =====================================================

        if "analysis" in st.session_state:
            analysis = st.session_state["analysis"]

            st.subheader("Feature Engineering")
            if apply_feature_engineering_enabled:
                engineered_df = apply_feature_engineering(
                    cleaned_df.copy(),
                    analysis
                )

                feature_steps = analysis.get(
                    "feature_engineering",
                    []
                )

                if len(feature_steps) > 0:
                    st.success(
                        f"Applied {len(feature_steps)} feature engineering step(s)."
                    )
                else:
                    st.info(
                        "No feature engineering steps were suggested, so the dataset was passed through unchanged."
                    )
            else:
                engineered_df = cleaned_df.copy()
                st.info(
                    "Feature engineering is disabled. Using the cleaned dataset as-is."
                )

            st.write(f"Features After Engineering: {engineered_df.shape[1]}")
            st.subheader("Engineered Dataset Preview")
            st.dataframe(engineered_df.head())

            # ==========================================
            # PREPROCESSING
            # ==========================================

            st.subheader("Preprocessing")
            preprocess_result = preprocess_dataset(
                engineered_df,
                target_column,
            )
            if isinstance(preprocess_result, (tuple, list)):
                if len(preprocess_result) == 4:
                    X_processed, y, preprocessor, label_encoder = preprocess_result
                elif len(preprocess_result) == 3:
                    X_processed, y, preprocessor = preprocess_result
                    label_encoder = None
                else:
                    raise ValueError(
                        "preprocess_dataset returned unexpected number of values."
                    )
            else:
                raise ValueError(
                    "preprocess_dataset must return a tuple/list."
                )

            st.success(f"Processed Shape: {X_processed.shape}")
            st.write("Processed Features")
            st.dataframe(X_processed.head())

            raw_X = engineered_df.drop(columns=[target_column], errors="ignore")
            raw_categorical = raw_X.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()
            if len(raw_categorical) > 0:
                st.caption(
                    f"One-hot encoding applied for categorical columns: {raw_categorical}"
                )

            # ==========================================
            # PROBLEM TYPE
            # ==========================================

            problem_type, problem_type_source = resolve_problem_type(
                y,
                analysis.get("problem_type", ""),
                problem_type_choice,
            )

            st.subheader("Problem Type")
            st.info(
                f"{problem_type} (source: {problem_type_source})"
            )

            # ==========================================
            # IMBALANCE CHECK
            # ==========================================

            imbalance_info = None

            if problem_type == "classification":
                imbalance_info = check_class_imbalance(y)
                st.subheader("Class Imbalance Analysis")
                st.json(imbalance_info)

            use_smote = (
                should_apply_smote(imbalance_strategy, imbalance_info)
                if problem_type == "classification"
                else False
            )
            use_class_weights = get_class_weight_flag(imbalance_strategy)

            if use_smote:
                st.warning("SMOTE will be applied inside CV folds.")

            if use_class_weights:
                st.warning("Class Weights Enabled.")

            # ==========================================
            # FEATURE IMPORTANCE
            # ==========================================

            st.subheader("Feature Importance Ranking")
            importance_df = feature_importance_ranking(X_processed, y, problem_type)
            st.dataframe(importance_df.head(20))

            # ==========================================
            # FEATURE SELECTION
            # ==========================================

            st.subheader("Feature Selection")

            if feature_selection_method == "Mutual Information":
                X_processed = mutual_information_selection(
                    X_processed,
                    y,
                    problem_type,
                    k=num_features,
                )

                st.success(
                    f"Mutual Information selected {X_processed.shape[1]} features."
                )

            elif feature_selection_method == "RFE":
                X_processed = rfe_selection(
                    X_processed,
                    y,
                    problem_type,
                    n_features=num_features,
                )

                st.success(f"RFE selected {X_processed.shape[1]} features.")

            else:
                st.info("Feature Selection Skipped")

            st.write("Selected Features")
            st.dataframe(X_processed.head())
            st.write(f"Final Feature Count: {X_processed.shape[1]}")

            selected_feature_columns = list(
                X_processed.columns
            )

            # ==========================================
            # DIMENSIONALITY REDUCTION
            # ==========================================

            st.subheader("Dimensionality Reduction")

            dimensionality_model = None
            final_feature_columns = list(
                X_processed.columns
            )

            if dimensionality_method == "PCA":
                X_processed, pca_model = apply_pca(X_processed, n_components)
                dimensionality_model = pca_model
                final_feature_columns = list(
                    X_processed.columns
                )

                st.success(f"PCA Applied ({X_processed.shape[1]} components)")

                variance_df = get_pca_variance(pca_model)
                st.subheader("PCA Variance Report")
                st.dataframe(variance_df)

            elif dimensionality_method == "ICA":
                X_processed, ica_model = apply_ica(X_processed, n_components)
                dimensionality_model = ica_model
                final_feature_columns = list(
                    X_processed.columns
                )

                st.success(f"ICA Applied ({X_processed.shape[1]} components)")

            else:
                st.info("Dimensionality Reduction Skipped")

            st.write("Final Training Matrix")
            st.dataframe(X_processed.head())
            st.write(f"Training Shape: {X_processed.shape}")

            # ==========================================
            # MODEL TRAINING
            # ==========================================

            st.subheader("Model Training")

            if st.button("Train Models", key="train_models"):
                with st.spinner("Training Models..."):
                    train_result = train_models(
                        X_processed,
                        y,
                        problem_type,
                        use_class_weights=use_class_weights,
                        use_smote=use_smote,
                    )
                    if isinstance(train_result, (tuple, list)):
                        if len(train_result) == 3:
                            leaderboard, trained_models, execution_report = train_result
                        elif len(train_result) == 2:
                            leaderboard, trained_models = train_result
                            execution_report = None
                        else:
                            raise ValueError(
                                "train_models returned unexpected number of values."
                            )
                    else:
                        raise ValueError(
                            "train_models must return a tuple/list."
                        )

                st.success("Training Complete")

                # ======================================
                # LEADERBOARD
                # ======================================

                st.subheader("Model Leaderboard")
                st.dataframe(leaderboard, use_container_width=True)

                # ======================================
                # BEST MODEL
                # ======================================

                st.subheader("Best Model")
                best_model_name = leaderboard.iloc[0]["Model"]

                st.success(f"Best Model: {best_model_name}")

                best_model = trained_models[best_model_name]
                st.write(best_model)

                # ======================================
                # BEST MODEL METRICS
                # ======================================

                st.subheader("Best Model Metrics")

                if problem_type == "classification":
                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        st.metric("Accuracy", leaderboard.iloc[0]["Accuracy Mean"])
                        st.metric("Precision", leaderboard.iloc[0]["Precision Mean"])

                    with metric_col2:
                        st.metric("Recall", leaderboard.iloc[0]["Recall Mean"])
                        st.metric("F1 Score", leaderboard.iloc[0]["F1 Mean"])

                    try:
                        roc_auc = leaderboard.iloc[0]["ROC AUC Mean"]

                        if roc_auc is not None:
                            st.metric("ROC AUC", roc_auc)

                    except Exception:
                        pass

                    st.subheader("Confusion Matrix")
                    try:
                        cv = StratifiedKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=42,
                        )
                        if best_model_name not in trained_models:
                            raise ValueError(
                                "Best model template not found for classification diagnostics."
                            )
                        best_model_template = trained_models[best_model_name]

                        y_pred_cv = cross_val_predict(
                            clone(best_model_template),
                            X_processed,
                            y,
                            cv=cv,
                            method="predict",
                        )

                        labels = np.unique(y)
                        cm = confusion_matrix(
                            y,
                            y_pred_cv,
                            labels=labels,
                        )
                        cm_df = pd.DataFrame(
                            cm,
                            index=[f"Actual_{label}" for label in labels],
                            columns=[f"Pred_{label}" for label in labels],
                        )
                        st.dataframe(cm_df, use_container_width=True)

                        st.subheader("Classification Report")
                        report_dict = classification_report(
                            y,
                            y_pred_cv,
                            output_dict=True,
                            zero_division=0,
                        )
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df, use_container_width=True)

                        st.subheader("Micro / Macro / Weighted Scores")
                        score_rows = [
                            {
                                "Average": "micro",
                                "Precision": precision_score(
                                    y,
                                    y_pred_cv,
                                    average="micro",
                                    zero_division=0,
                                ),
                                "Recall": recall_score(
                                    y,
                                    y_pred_cv,
                                    average="micro",
                                    zero_division=0,
                                ),
                                "F1": f1_score(
                                    y,
                                    y_pred_cv,
                                    average="micro",
                                    zero_division=0,
                                ),
                            },
                            {
                                "Average": "macro",
                                "Precision": precision_score(
                                    y,
                                    y_pred_cv,
                                    average="macro",
                                    zero_division=0,
                                ),
                                "Recall": recall_score(
                                    y,
                                    y_pred_cv,
                                    average="macro",
                                    zero_division=0,
                                ),
                                "F1": f1_score(
                                    y,
                                    y_pred_cv,
                                    average="macro",
                                    zero_division=0,
                                ),
                            },
                            {
                                "Average": "weighted",
                                "Precision": precision_score(
                                    y,
                                    y_pred_cv,
                                    average="weighted",
                                    zero_division=0,
                                ),
                                "Recall": recall_score(
                                    y,
                                    y_pred_cv,
                                    average="weighted",
                                    zero_division=0,
                                ),
                                "F1": f1_score(
                                    y,
                                    y_pred_cv,
                                    average="weighted",
                                    zero_division=0,
                                ),
                            },
                        ]
                        scores_df = pd.DataFrame(score_rows)
                        st.dataframe(scores_df, use_container_width=True)

                    except Exception as diagnostics_error:
                        st.warning(
                            f"Could not generate classification diagnostics: {diagnostics_error}"
                        )

                else:
                    metric_rows = [
                        [
                            ("R2 Score", "R2 Mean"),
                            ("MAE", "MAE Mean"),
                            ("RMSE", "RMSE Mean"),
                        ],
                        [
                            ("MSE", "MSE Mean"),
                            ("Median AE", "MedAE Mean"),
                            ("MAPE", "MAPE Mean (%)"),
                        ],
                        [
                            ("Explained Variance", "Explained Variance Mean"),
                            ("Max Error", "Max Error Mean"),
                            (None, None),
                        ],
                    ]

                    for row in metric_rows:
                        cols = st.columns(3)
                        for idx, metric_item in enumerate(row):
                            label, key_name = metric_item
                            if label is None:
                                continue
                            with cols[idx]:
                                st.metric(
                                    label,
                                    leaderboard.iloc[0][key_name]
                                )

                # ======================================
                # DOWNLOAD LEADERBOARD
                # ======================================

                st.subheader("Download Results")
                leaderboard_csv = leaderboard.to_csv(index=False)

                st.download_button(
                    label="Download Leaderboard",
                    data=leaderboard_csv,
                    file_name="leaderboard.csv",
                    mime="text/csv",
                )

                # ======================================
                # SESSION STORAGE
                # ======================================

                st.session_state["leaderboard"] = leaderboard
                st.session_state["best_model"] = best_model_name
                st.session_state["problem_type"] = problem_type
                st.session_state["best_model_bundle"] = build_model_bundle(
                    best_model_name=best_model_name,
                    best_model=best_model,
                    preprocessor=preprocessor,
                    label_encoder=label_encoder,
                    target_column=target_column,
                    problem_type=problem_type,
                    analysis=analysis,
                    raw_feature_columns=[
                        col for col in cleaned_df.columns
                        if col != target_column
                    ],
                    preprocessor_feature_columns=list(
                        getattr(
                            preprocessor,
                            "feature_names_in_",
                            engineered_df.drop(
                                columns=[
                                    target_column
                                ],
                                errors="ignore"
                            ).columns
                        )
                    ),
                    selected_feature_columns=selected_feature_columns,
                    final_feature_columns=final_feature_columns,
                    feature_selection_method=feature_selection_method,
                    dimensionality_method=dimensionality_method,
                    dimensionality_model=dimensionality_model,
                    apply_feature_engineering_enabled=apply_feature_engineering_enabled,
                    columns_to_drop=columns_to_drop,
                    description=description,
                    cleaned_df=cleaned_df
                )

            if (
                st.session_state.get("best_model_bundle") is not None
                and st.session_state["best_model_bundle"].get(
                    "training_signature"
                )
                == build_analysis_cache_key(
                    cleaned_df,
                    target_column,
                    description
                )
            ):
                best_model_bundle = st.session_state["best_model_bundle"]

                st.subheader("Best Model Download")
                bundle_name = (
                    best_model_bundle["best_model_name"]
                    .lower()
                    .replace(" ", "_")
                )
                bundle_bytes = pickle.dumps(
                    best_model_bundle
                )
                st.download_button(
                    label="Download Best Model Bundle",
                    data=bundle_bytes,
                    file_name=f"{bundle_name}_bundle.pkl",
                    mime="application/octet-stream",
                )

                st.subheader("Test Prediction")
                prediction_tabs = st.tabs(
                    [
                        "CSV Upload",
                        "Single Value"
                    ]
                )

                with prediction_tabs[0]:
                    st.caption(
                        "Upload a CSV or Excel file containing the feature columns for prediction. "
                        "If the target column is present, it will be ignored."
                    )
                    prediction_file = st.file_uploader(
                        "Upload Test File",
                        type=["csv", "xlsx", "xls"],
                        key="prediction_test_file"
                    )

                    if prediction_file is not None:
                        if st.button(
                            "Run CSV Prediction",
                            key="run_csv_prediction"
                        ):
                            try:
                                prediction_input_df = load_tabular_file(
                                    prediction_file
                                )
                                prediction_output_df = generate_predictions(
                                    prediction_input_df,
                                    best_model_bundle
                                )
                                st.success("Predictions generated successfully.")
                                st.dataframe(
                                    prediction_output_df,
                                    use_container_width=True
                                )
                                st.download_button(
                                    label="Download Predictions",
                                    data=prediction_output_df.to_csv(
                                        index=False
                                    ),
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                            except Exception as prediction_error:
                                st.error(
                                    f"Prediction failed: {prediction_error}"
                                )

                with prediction_tabs[1]:
                    st.caption(
                        "Enter one row of feature values and run a single prediction."
                    )
                    with st.form("single_prediction_form"):
                        single_prediction_values = render_single_prediction_inputs(
                            best_model_bundle["raw_feature_columns"],
                            cleaned_df
                        )
                        single_prediction_submit = st.form_submit_button(
                            "Predict Single Row"
                        )

                    if single_prediction_submit:
                        try:
                            single_prediction_df = pd.DataFrame(
                                [
                                    single_prediction_values
                                ]
                            )
                            prediction_output_df = generate_predictions(
                                single_prediction_df,
                                best_model_bundle
                            )
                            st.success("Prediction generated successfully.")
                            st.dataframe(
                                prediction_output_df,
                                use_container_width=True
                            )
                        except Exception as prediction_error:
                            st.error(
                                f"Prediction failed: {prediction_error}"
                            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())
