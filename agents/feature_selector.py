import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


# =====================================================
# MUTUAL INFORMATION
# =====================================================

def mutual_information_selection(
        X,
        y,
        problem_type,
        k=10
):

    k = min(
        k,
        X.shape[1]
    )

    if problem_type == "classification":

        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=k
        )

    else:

        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=k
        )

    selector.fit(
        X,
        y
    )

    selected_columns = (
        X.columns[
            selector.get_support()
        ]
    )

    X_selected = (
        X[
            selected_columns
        ]
        .copy()
    )

    return X_selected


# =====================================================
# MUTUAL INFORMATION RANKING
# =====================================================

def mutual_information_ranking(
        X,
        y,
        problem_type
):

    if problem_type == "classification":

        scores = (
            mutual_info_classif(
                X,
                y,
                random_state=42
            )
        )

    else:

        scores = (
            mutual_info_regression(
                X,
                y,
                random_state=42
            )
        )

    importance_df = pd.DataFrame(
        {

            "Feature":
            X.columns,

            "Importance":
            scores
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return importance_df


# =====================================================
# RFE
# =====================================================

def rfe_selection(
        X,
        y,
        problem_type,
        n_features=10
):

    n_features = min(
        n_features,
        X.shape[1]
    )

    if problem_type == "classification":

        estimator = LogisticRegression(
            max_iter=3000
        )

    else:

        estimator = LinearRegression()

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features
    )

    selector.fit(
        X,
        y
    )

    selected_columns = (
        X.columns[
            selector.support_
        ]
    )

    X_selected = (
        X[
            selected_columns
        ]
        .copy()
    )

    return X_selected


# =====================================================
# RFE RANKING
# =====================================================

def rfe_ranking(
        X,
        y,
        problem_type
):

    if problem_type == "classification":

        estimator = LogisticRegression(
            max_iter=3000
        )

    else:

        estimator = LinearRegression()

    selector = RFE(
        estimator=estimator,
        n_features_to_select=1
    )

    selector.fit(
        X,
        y
    )

    ranking_df = pd.DataFrame(
        {

            "Feature":
            X.columns,

            "Rank":
            selector.ranking_
        }
    )

    ranking_df = (
        ranking_df
        .sort_values(
            by="Rank"
        )
        .reset_index(
            drop=True
        )
    )

    return ranking_df


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def feature_importance_ranking(
        X,
        y,
        problem_type
):

    if problem_type == "classification":

        model = (
            RandomForestClassifier(
                n_estimators=300,
                random_state=42
            )
        )

    else:

        model = (
            RandomForestRegressor(
                n_estimators=300,
                random_state=42
            )
        )

    model.fit(
        X,
        y
    )

    importance_df = pd.DataFrame(
        {

            "Feature":
            X.columns,

            "Importance":
            model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return importance_df


# =====================================================
# FEATURE SELECTION REPORT
# =====================================================

def generate_feature_selection_report(
        X,
        y,
        problem_type
):

    mi_report = (
        mutual_information_ranking(
            X,
            y,
            problem_type
        )
    )

    rfe_report = (
        rfe_ranking(
            X,
            y,
            problem_type
        )
    )

    rf_report = (
        feature_importance_ranking(
            X,
            y,
            problem_type
        )
    )

    return {

        "mutual_information":
        mi_report,

        "rfe":
        rfe_report,

        "feature_importance":
        rf_report
    }