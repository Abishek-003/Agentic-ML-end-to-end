import pandas as pd

from collections import Counter

from imblearn.over_sampling import (
    SMOTE
)


def is_classification_problem(
        y
):

    if (
        y.dtype == "object"
    ):
        return True

    if (
        pd.Series(y)
        .nunique()
        <= 20
    ):
        return True

    return False


def check_class_imbalance(
        y,
        threshold=0.20
):

    if not is_classification_problem(
        pd.Series(y)
    ):

        return {
            "is_classification": False,
            "is_imbalanced": False,
            "imbalance_ratio": None,
            "class_distribution": None
        }

    class_counts = Counter(y)

    total_samples = sum(
        class_counts.values()
    )

    class_distribution = {

        str(label):
        round(
            count / total_samples,
            4
        )

        for label, count
        in class_counts.items()
    }

    minority_count = min(
        class_counts.values()
    )

    majority_count = max(
        class_counts.values()
    )

    imbalance_ratio = round(
        minority_count
        /
        majority_count,
        4
    )

    is_imbalanced = (
        imbalance_ratio
        <
        threshold
    )

    return {

        "is_classification":
        True,

        "is_imbalanced":
        is_imbalanced,

        "imbalance_ratio":
        imbalance_ratio,

        "class_distribution":
        class_distribution,

        "minority_class_size":
        minority_count,

        "majority_class_size":
        majority_count
    }


def apply_smote(
        X,
        y,
        random_state=42
):

    smote = SMOTE(
        random_state=random_state
    )

    X_resampled, y_resampled = (
        smote.fit_resample(
            X,
            y
        )
    )

    return (
        X_resampled,
        y_resampled
    )


def should_apply_smote(
        strategy,
        imbalance_info
):

    if (
        imbalance_info is None
    ):
        return False

    if (
        not imbalance_info.get(
            "is_classification",
            False
        )
    ):
        return False

    if strategy == "SMOTE":

        return True

    if strategy == "Auto":

        return (
            imbalance_info.get(
                "is_imbalanced",
                False
            )
        )

    return False


def get_class_weight_flag(
        strategy
):

    if strategy == "Class Weights":

        return True

    return False


def get_recommended_strategy(
        imbalance_info
):

    if (
        imbalance_info is None
    ):

        return "None"

    if (
        not imbalance_info.get(
            "is_classification",
            False
        )
    ):

        return "None"

    ratio = (
        imbalance_info.get(
            "imbalance_ratio",
            1.0
        )
    )

    if ratio < 0.10:

        return "SMOTE"

    if ratio < 0.25:

        return "Class Weights"

    return "None"


def generate_imbalance_report(
        y
):

    info = (
        check_class_imbalance(
            y
        )
    )

    recommendation = (
        get_recommended_strategy(
            info
        )
    )

    return {

        "analysis":
        info,

        "recommended_strategy":
        recommendation
    }