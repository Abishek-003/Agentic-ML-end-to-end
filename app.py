import traceback
import json

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
    cache_version = "v2_problem_type_fix"
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
    unique_ratio = unique_count / total_count

    if (
        pd.api.types.is_object_dtype(y_non_null)
        or pd.api.types.is_categorical_dtype(y_non_null)
        or pd.api.types.is_bool_dtype(y_non_null)
    ):
        return "classification", "heuristic_categorical"

    if pd.api.types.is_integer_dtype(y_non_null):
        if unique_ratio < 0.98:
            return "classification", "heuristic_integer_discrete"
        return "regression", "heuristic_integer_continuous"

    if pd.api.types.is_float_dtype(y_non_null):
        is_integer_like = np.all(
            np.isclose(
                y_non_null.values,
                np.round(y_non_null.values),
                atol=1e-9,
            )
        )
        if is_integer_like and unique_ratio < 0.98:
            return "classification", "heuristic_float_integer_like"
        if unique_count <= max(20, int(total_count * 0.05)):
            return "classification", "heuristic_float_low_cardinality"
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

    # Prefer heuristic when target looks strongly discrete but LLM says regression.
    if (
        heuristic_type == "classification"
        and llm_type == "regression"
    ):
        return "classification", f"{heuristic_reason}_override_llm"

    return llm_type, "llm"


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="Agentic ML Pipeline", layout="wide")

st.title("Agentic ML Pipeline")
st.markdown(
    """
Upload dataset -> Analyze dataset -> Feature Engineering ->
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
)

feature_selection_method = st.sidebar.selectbox(
    "Feature Selection",
    ["None", "Mutual Information", "RFE"],
)

num_features = st.sidebar.number_input(
    "Features To Keep",
    min_value=1,
    max_value=1000,
    value=10,
)

dimensionality_method = st.sidebar.selectbox(
    "Dimensionality Reduction",
    ["None", "PCA", "ICA"],
)

n_components = st.sidebar.slider(
    "Number of Components",
    min_value=2,
    max_value=50,
    value=5,
)

imbalance_strategy = st.sidebar.selectbox(
    "Imbalance Handling",
    ["Auto", "None", "Class Weights", "SMOTE"],
)

run_llm_cleaner = st.sidebar.checkbox("Run LLM Feature Cleaner", value=False)


# =====================================================
# DATASET INPUT
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"],
)

description = st.text_area(
    "Dataset Description",
    placeholder="Describe the ML problem...",
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
        target_column = st.selectbox("Select Target Column", df.columns.tolist())

        # ==========================================
        # FEATURE CLEANER
        # ==========================================

        st.subheader("Feature Cleaner Suggestions")
        feature_report = analyze_features(df, target_column)
        st.json(feature_report["analysis"])

        columns_to_drop = st.multiselect(
            "Columns To Drop",
            feature_report["suggested_drop_columns"],
            default=[],
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

        if st.button("Analyze Dataset"):
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
                analysis = analyze_dataset(metadata, description, target_column)
                analysis = canonicalize_analysis(analysis)
                analysis_cache[cache_key] = analysis

            st.session_state["analysis"] = analysis
            st.success("Dataset Analysis Complete")
            st.json(analysis)

        # =====================================================
        # PIPELINE EXECUTION
        # =====================================================

        if "analysis" in st.session_state:
            analysis = st.session_state["analysis"]

            st.subheader("Feature Engineering")
            engineered_df = apply_feature_engineering(cleaned_df.copy(), analysis)

            st.success(f"Features After Engineering: {engineered_df.shape[1]}")
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

            # ==========================================
            # DIMENSIONALITY REDUCTION
            # ==========================================

            st.subheader("Dimensionality Reduction")

            if dimensionality_method == "PCA":
                X_processed, pca_model = apply_pca(X_processed, n_components)

                st.success(f"PCA Applied ({X_processed.shape[1]} components)")

                variance_df = get_pca_variance(pca_model)
                st.subheader("PCA Variance Report")
                st.dataframe(variance_df)

            elif dimensionality_method == "ICA":
                X_processed, ica_model = apply_ica(X_processed, n_components)

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

            if st.button("Train Models"):
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
                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        st.metric("R2 Score", leaderboard.iloc[0]["R2 Mean"])
                        st.metric("MAE", leaderboard.iloc[0]["MAE Mean"])

                    with metric_col2:
                        st.metric("RMSE", leaderboard.iloc[0]["RMSE Mean"])

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

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())
