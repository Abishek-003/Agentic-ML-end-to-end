import pandas as pd


SUPPORTED_FILE_TYPES = [
    ".csv",
    ".xlsx",
    ".xls"
]


def load_csv(file):

    return pd.read_csv(
        file
    )


def load_excel(file):

    return pd.read_excel(
        file
    )


def validate_dataset(df):

    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    if len(df.columns) < 2:

        raise ValueError(
            "Dataset must contain at least 2 columns."
        )

    return True


def load_dataset(uploaded_file):

    if uploaded_file is None:

        raise ValueError(
            "No file uploaded."
        )

    filename = (
        uploaded_file.name
        .lower()
    )

    if filename.endswith(
        ".csv"
    ):

        df = load_csv(
            uploaded_file
        )

    elif (
        filename.endswith(".xlsx")
        or
        filename.endswith(".xls")
    ):

        df = load_excel(
            uploaded_file
        )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Upload CSV or Excel."
        )

    validate_dataset(df)

    return df


def get_dataset_summary(df):

    return {

        "rows":
        int(
            df.shape[0]
        ),

        "columns":
        int(
            df.shape[1]
        ),

        "column_names":
        list(
            df.columns
        ),

        "missing_values":
        (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "data_types":
        (
            df.dtypes
            .astype(str)
            .to_dict()
        )
    }