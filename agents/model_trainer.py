import pandas as pd
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    KFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)

from sklearn.naive_bayes import (
    GaussianNB
)

from imblearn.over_sampling import (
    SMOTE
)

try:
    from xgboost import (
        XGBClassifier,
        XGBRegressor
    )
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import (
        LGBMClassifier,
        LGBMRegressor
    )
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None


# =====================================================
# CLASSIFICATION MODELS
# =====================================================

def get_classification_models(
        use_class_weights=False
):

    class_weight = (
        "balanced"
        if use_class_weights
        else None
    )

    models = {

        "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            class_weight=class_weight
        ),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=class_weight
        ),

        "SVM":
        SVC(
            probability=True,
            class_weight=class_weight
        ),

        "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight=class_weight
        ),

        "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        ),

        "AdaBoost":
        AdaBoostClassifier(
            random_state=42
        ),

        "KNN":
        KNeighborsClassifier(
            n_neighbors=7
        ),

        "Gaussian NB":
        GaussianNB()
    }

    if XGBClassifier is not None:
        models[
            "XGBoost"
        ] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )

    if LGBMClassifier is not None:
        models[
            "LightGBM"
        ] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    return models


# =====================================================
# REGRESSION MODELS
# =====================================================

def get_regression_models():

    models = {

        "Linear Regression":
        LinearRegression(),

        "Ridge":
        Ridge(
            alpha=1.0
        ),

        "Lasso":
        Lasso(
            alpha=0.001,
            max_iter=5000,
            random_state=42
        ),

        "Elastic Net":
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=5000,
            random_state=42
        ),

        "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

        "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=42
        ),

        "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

        "AdaBoost":
        AdaBoostRegressor(
            random_state=42
        ),

        "KNN":
        KNeighborsRegressor(
            n_neighbors=7
        ),

        "SVR":
        SVR()
    }

    if XGBRegressor is not None:
        models[
            "XGBoost"
        ] = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )

    if LGBMRegressor is not None:
        models[
            "LightGBM"
        ] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    return models


# =====================================================
# CLASSIFICATION CV
# =====================================================

def train_classification_models(
        X,
        y,
        use_class_weights=False,
        use_smote=False,
        n_splits=5
):

    models = (
        get_classification_models(
            use_class_weights
        )
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    leaderboard = []

    trained_models = {}

    for model_name, model in models.items():

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []

        for train_idx, test_idx in cv.split(X, y):

            X_train = (
                X.iloc[train_idx]
            )

            X_test = (
                X.iloc[test_idx]
            )

            y_train = (
                y.iloc[train_idx]
            )

            y_test = (
                y.iloc[test_idx]
            )

            # ------------------
            # SMOTE INSIDE FOLD
            # ------------------

            if use_smote:

                smote = SMOTE(
                    random_state=42
                )

                X_train, y_train = (
                    smote.fit_resample(
                        X_train,
                        y_train
                    )
                )

            model.fit(
                X_train,
                y_train
            )

            predictions = (
                model.predict(
                    X_test
                )
            )

            accuracy_scores.append(
                accuracy_score(
                    y_test,
                    predictions
                )
            )

            precision_scores.append(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            )

            recall_scores.append(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
            )

            f1_scores.append(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted"
                )
            )

            try:

                if len(
                    np.unique(y)
                ) == 2:

                    probabilities = (
                        model.predict_proba(
                            X_test
                        )[:, 1]
                    )

                    auc_scores.append(
                        roc_auc_score(
                            y_test,
                            probabilities
                        )
                    )

            except Exception:
                pass

        leaderboard.append(
            {

                "Model":
                model_name,

                "Accuracy Mean":
                round(
                    np.mean(
                        accuracy_scores
                    ),
                    4
                ),

                "Accuracy Std":
                round(
                    np.std(
                        accuracy_scores
                    ),
                    4
                ),

                "F1 Mean":
                round(
                    np.mean(
                        f1_scores
                    ),
                    4
                ),

                "F1 Std":
                round(
                    np.std(
                        f1_scores
                    ),
                    4
                ),

                "Precision Mean":
                round(
                    np.mean(
                        precision_scores
                    ),
                    4
                ),

                "Recall Mean":
                round(
                    np.mean(
                        recall_scores
                    ),
                    4
                ),

                "ROC AUC Mean":
                (
                    round(
                        np.mean(
                            auc_scores
                        ),
                        4
                    )
                    if len(
                        auc_scores
                    ) > 0
                    else None
                )
            }
        )

        trained_models[
            model_name
        ] = model

    leaderboard = (
        pd.DataFrame(
            leaderboard
        )
        .sort_values(
            by="Accuracy Mean",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return (
        leaderboard,
        trained_models
    )


# =====================================================
# REGRESSION CV
# =====================================================

def train_regression_models(
        X,
        y,
        n_splits=5
):

    models = (
        get_regression_models()
    )

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    leaderboard = []

    trained_models = {}

    for model_name, model in models.items():

        r2_scores = []
        mae_scores = []
        rmse_scores = []

        for train_idx, test_idx in cv.split(X):

            X_train = (
                X.iloc[train_idx]
            )

            X_test = (
                X.iloc[test_idx]
            )

            y_train = (
                y.iloc[train_idx]
            )

            y_test = (
                y.iloc[test_idx]
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = (
                model.predict(
                    X_test
                )
            )

            r2_scores.append(
                r2_score(
                    y_test,
                    predictions
                )
            )

            mae_scores.append(
                mean_absolute_error(
                    y_test,
                    predictions
                )
            )

            rmse_scores.append(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )
            )

        leaderboard.append(
            {

                "Model":
                model_name,

                "R2 Mean":
                round(
                    np.mean(
                        r2_scores
                    ),
                    4
                ),

                "R2 Std":
                round(
                    np.std(
                        r2_scores
                    ),
                    4
                ),

                "MAE Mean":
                round(
                    np.mean(
                        mae_scores
                    ),
                    4
                ),

                "RMSE Mean":
                round(
                    np.mean(
                        rmse_scores
                    ),
                    4
                )
            }
        )

        trained_models[
            model_name
        ] = model

    leaderboard = (
        pd.DataFrame(
            leaderboard
        )
        .sort_values(
            by="R2 Mean",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return (
        leaderboard,
        trained_models
    )


# =====================================================
# MAIN ROUTER
# =====================================================

def train_models(
        X,
        y,
        problem_type,
        use_class_weights=False,
        use_smote=False
):

    if (
        problem_type
        ==
        "classification"
    ):

        return (
            train_classification_models(
                X,
                y,
                use_class_weights=use_class_weights,
                use_smote=use_smote
            )
        )

    return (
        train_regression_models(
            X,
            y
        )
    )
