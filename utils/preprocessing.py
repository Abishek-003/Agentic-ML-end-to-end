import pandas as pd

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)


def detect_feature_types(
        X
):

    numerical_features = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    return (
        numerical_features,
        categorical_features
    )


def build_preprocessor(
        numerical_features,
        categorical_features
):

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numeric",
                numeric_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],

        remainder="drop"
    )

    return preprocessor


def encode_target(y):

    label_encoder = None

    if (
        y.dtype == "object"
        or
        str(y.dtype) == "category"
        or
        str(y.dtype) == "bool"
    ):

        label_encoder = (
            LabelEncoder()
        )

        y = (
            label_encoder
            .fit_transform(y)
        )

    return (
        y,
        label_encoder
    )


def get_feature_names(
        preprocessor,
        numerical_features,
        categorical_features
):

    feature_names = []

    feature_names.extend(
        numerical_features
    )

    if len(
        categorical_features
    ) > 0:

        encoder = (
            preprocessor
            .named_transformers_[
                "categorical"
            ]
            .named_steps[
                "encoder"
            ]
        )

        encoded_columns = (
            encoder
            .get_feature_names_out(
                categorical_features
            )
        )

        feature_names.extend(
            encoded_columns
        )

    return feature_names


def preprocess_dataset(
        df,
        target_column
):

    if target_column not in df.columns:

        raise ValueError(
            f"{target_column} not found in dataset."
        )

    X = df.drop(
        columns=[
            target_column
        ]
    )

    y = df[
        target_column
    ]

    (
        numerical_features,
        categorical_features
    ) = detect_feature_types(
        X
    )

    preprocessor = (
        build_preprocessor(
            numerical_features,
            categorical_features
        )
    )

    X_processed = (
        preprocessor
        .fit_transform(X)
    )

    (
        y,
        label_encoder
    ) = encode_target(
        y
    )

    feature_names = (
        get_feature_names(
            preprocessor,
            numerical_features,
            categorical_features
        )
    )

    X_processed = pd.DataFrame(
        X_processed,
        columns=feature_names
    )

    return (
        X_processed,
        pd.Series(y),
        preprocessor,
        label_encoder
    )