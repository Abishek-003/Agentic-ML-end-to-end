import numpy as np
import pandas as pd


def is_numeric_column(
        df,
        column_name
):

    if column_name not in df.columns:
        return False

    return pd.api.types.is_numeric_dtype(
        df[column_name]
    )


def safe_divide(
        numerator,
        denominator
):

    denominator = denominator.replace(
        0,
        np.nan
    )

    return numerator / denominator


def create_ratio_feature(
        df,
        columns,
        feature_name
):

    if len(columns) != 2:
        return df

    col1, col2 = columns

    if not (
        is_numeric_column(
            df,
            col1
        )
        and
        is_numeric_column(
            df,
            col2
        )
    ):
        return df

    try:

        df[
            feature_name
        ] = safe_divide(
            df[col1],
            df[col2]
        )

    except Exception:

        pass

    return df


def create_interaction_feature(
        df,
        columns,
        feature_name
):

    if len(columns) != 2:
        return df

    col1, col2 = columns

    if not (
        is_numeric_column(
            df,
            col1
        )
        and
        is_numeric_column(
            df,
            col2
        )
    ):
        return df

    try:

        df[
            feature_name
        ] = (
            df[col1]
            *
            df[col2]
        )

    except Exception:

        pass

    return df


def create_sum_feature(
        df,
        columns,
        feature_name
):

    if len(columns) < 2:
        return df

    for col in columns:

        if not is_numeric_column(
            df,
            col
        ):
            return df

    try:

        df[
            feature_name
        ] = (
            df[columns]
            .sum(axis=1)
        )

    except Exception:

        pass

    return df


def create_difference_feature(
        df,
        columns,
        feature_name
):

    if len(columns) != 2:
        return df

    col1, col2 = columns

    if not (
        is_numeric_column(
            df,
            col1
        )
        and
        is_numeric_column(
            df,
            col2
        )
    ):
        return df

    try:

        df[
            feature_name
        ] = (
            df[col1]
            -
            df[col2]
        )

    except Exception:

        pass

    return df


def create_log_feature(
        df,
        columns,
        feature_name
):

    if len(columns) != 1:
        return df

    col = columns[0]

    if not is_numeric_column(
        df,
        col
    ):
        return df

    try:

        values = (
            df[col]
            .copy()
        )

        values = values.clip(
            lower=0
        )

        df[
            feature_name
        ] = np.log1p(
            values
        )

    except Exception:

        pass

    return df


def create_square_feature(
        df,
        columns,
        feature_name
):

    if len(columns) != 1:
        return df

    col = columns[0]

    if not is_numeric_column(
        df,
        col
    ):
        return df

    try:

        df[
            feature_name
        ] = (
            df[col] ** 2
        )

    except Exception:

        pass

    return df


def create_sqrt_feature(
        df,
        columns,
        feature_name
):

    if len(columns) != 1:
        return df

    col = columns[0]

    if not is_numeric_column(
        df,
        col
    ):
        return df

    try:

        values = (
            df[col]
            .clip(lower=0)
        )

        df[
            feature_name
        ] = np.sqrt(
            values
        )

    except Exception:

        pass

    return df


def apply_single_feature_operation(
        df,
        feature
):

    operation = (
        feature.get(
            "operation",
            ""
        )
        .lower()
        .strip()
    )

    columns = feature.get(
        "columns",
        []
    )

    feature_name = (
        feature.get(
            "new_feature_name",
            f"{operation}_feature"
        )
    )

    if feature_name in df.columns:

        return df

    if operation == "ratio":

        return create_ratio_feature(
            df,
            columns,
            feature_name
        )

    elif operation == "interaction":

        return create_interaction_feature(
            df,
            columns,
            feature_name
        )

    elif operation == "sum":

        return create_sum_feature(
            df,
            columns,
            feature_name
        )

    elif operation == "difference":

        return create_difference_feature(
            df,
            columns,
            feature_name
        )

    elif operation == "log_transform":

        return create_log_feature(
            df,
            columns,
            feature_name
        )

    elif operation == "square":

        return create_square_feature(
            df,
            columns,
            feature_name
        )

    elif operation == "sqrt":

        return create_sqrt_feature(
            df,
            columns,
            feature_name
        )

    return df


def apply_feature_engineering(
        df,
        analysis
):

    feature_list = (
        analysis.get(
            "feature_engineering",
            []
        )
    )

    if len(
        feature_list
    ) == 0:

        return df

    for feature in feature_list:

        try:

            df = (
                apply_single_feature_operation(
                    df,
                    feature
                )
            )

        except Exception as e:

            print(
                f"Feature engineering failed: {e}"
            )

    return df