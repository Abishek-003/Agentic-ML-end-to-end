import pandas as pd


ID_KEYWORDS = [
    "id",
    "customerid",
    "customer_id",
    "userid",
    "user_id",
    "employeeid",
    "employee_id",
    "transactionid",
    "transaction_id",
    "recordid",
    "record_id",
    "uuid"
]


def detect_id_columns(df):

    results = []

    for col in df.columns:

        col_lower = col.lower()

        unique_ratio = (
            df[col].nunique(dropna=True)
            /
            max(
                len(df),
                1
            )
        )

        if any(
            keyword in col_lower
            for keyword in ID_KEYWORDS
        ):

            results.append(
                (
                    col,
                    "Identifier Column"
                )
            )

            continue

        if (
            unique_ratio > 0.98
            and
            (
                pd.api.types.is_integer_dtype(
                    df[col]
                )
                or
                pd.api.types.is_object_dtype(
                    df[col]
                )
            )
        ):

            results.append(
                (
                    col,
                    "High Uniqueness"
                )
            )

    return results


def detect_constant_columns(df):

    results = []

    for col in df.columns:

        if (
            df[col]
            .nunique(
                dropna=False
            )
            <= 1
        ):

            results.append(
                (
                    col,
                    "Constant Column"
                )
            )

    return results


def detect_near_constant_columns(
        df,
        threshold=0.99
):

    results = []

    for col in df.columns:

        try:

            dominant_ratio = (
                df[col]
                .value_counts(
                    normalize=True,
                    dropna=False
                )
                .max()
            )

            if dominant_ratio > threshold:

                results.append(
                    (
                        col,
                        f"Near Constant ({dominant_ratio:.2f})"
                    )
                )

        except Exception:
            pass

    return results


def detect_high_missing_columns(
        df,
        threshold=0.80
):

    results = []

    for col in df.columns:

        missing_ratio = (
            df[col]
            .isnull()
            .mean()
        )

        if missing_ratio > threshold:

            results.append(
                (
                    col,
                    f"High Missing ({missing_ratio:.2f})"
                )
            )

    return results


def detect_duplicate_columns(df):

    results = []

    columns = list(
        df.columns
    )

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns)
        ):

            col1 = columns[i]
            col2 = columns[j]

            try:

                if (
                    df[col1]
                    .equals(
                        df[col2]
                    )
                ):

                    results.append(
                        (
                            col2,
                            f"Duplicate of {col1}"
                        )
                    )

            except Exception:
                pass

    return results


def detect_high_cardinality_columns(
        df,
        threshold=0.90
):

    results = []

    for col in df.columns:

        if not (
            pd.api.types.is_object_dtype(
                df[col]
            )
            or
            pd.api.types.is_string_dtype(
                df[col]
            )
        ):
            continue

        cardinality_ratio = (
            df[col]
            .nunique(
                dropna=True
            )
            /
            max(
                len(df),
                1
            )
        )

        if cardinality_ratio > threshold:

            results.append(
                (
                    col,
                    f"High Cardinality ({cardinality_ratio:.2f})"
                )
            )

    return results


def detect_datetime_columns(df):

    results = []

    for col in df.columns:

        try:

            converted = pd.to_datetime(
                df[col],
                errors="coerce"
            )

            valid_ratio = (
                converted
                .notna()
                .mean()
            )

            if valid_ratio > 0.90:

                results.append(
                    (
                        col,
                        "Datetime Candidate"
                    )
                )

        except Exception:
            pass

    return results


def analyze_features(
        df,
        target_column=None
):

    analysis = {

        "id_columns":
        detect_id_columns(df),

        "constant_columns":
        detect_constant_columns(df),

        "near_constant_columns":
        detect_near_constant_columns(df),

        "high_missing_columns":
        detect_high_missing_columns(df),

        "duplicate_columns":
        detect_duplicate_columns(df),

        "high_cardinality_columns":
        detect_high_cardinality_columns(df),

        "datetime_columns":
        detect_datetime_columns(df)
    }

    hard_drop_categories = [
        "id_columns",
        "constant_columns",
        "high_missing_columns",
        "duplicate_columns"
    ]

    review_only_categories = [
        "near_constant_columns",
        "high_cardinality_columns",
        "datetime_columns"
    ]

    suggested_drop_columns = []
    review_columns = []

    for category_name, category_values in analysis.items():

        for item in category_values:

            column_name = item[0]

            if (
                target_column
                and
                column_name == target_column
            ):
                continue

            if category_name in hard_drop_categories:
                suggested_drop_columns.append(column_name)
            elif category_name in review_only_categories:
                review_columns.append(column_name)

    suggested_drop_columns = list(
        sorted(
            set(
                suggested_drop_columns
            )
        )
    )

    review_columns = list(
        sorted(
            set(
                review_columns
            )
        )
    )

    return {

        "analysis":
        analysis,

        "suggested_drop_columns":
        suggested_drop_columns,

        "review_columns":
        review_columns
    }


def drop_selected_columns(
        df,
        columns_to_drop
):

    return df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )
