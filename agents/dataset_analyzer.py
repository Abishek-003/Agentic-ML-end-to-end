import json
import re

from utils.llm import (
    client,
    DEFAULT_MODEL
)


def generate_metadata(df):

    metadata = {

        "rows":
        int(
            len(df)
        ),

        "columns":
        int(
            len(df.columns)
        ),

        "column_names":
        list(
            df.columns
        ),

        "dtypes":
        (
            df.dtypes
            .astype(str)
            .to_dict()
        ),

        "nunique":
        (
            df.nunique(
                dropna=True
            )
            .to_dict()
        ),

        "missing_values":
        (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "sample_rows":
        (
            df.head(5)
            .to_dict()
        ),

        "summary_statistics":
        (
            df.describe(
                include="all"
            )
            .fillna("")
            .to_dict()
        )
    }

    return metadata


def extract_json(
        response_text
):
    cleaned_text = str(
        response_text or ""
    )

    cleaned_text = re.sub(
        r"<think>[\s\S]*?(</think>|$)",
        "",
        cleaned_text,
        flags=re.IGNORECASE
    )

    cleaned_text = cleaned_text.replace(
        "```json",
        ""
    )
    cleaned_text = cleaned_text.replace(
        "```JSON",
        ""
    )
    cleaned_text = cleaned_text.replace(
        "```",
        ""
    )

    start_index = cleaned_text.find("{")
    if start_index == -1:
        raise ValueError(
            f"No JSON found.\n\n{cleaned_text[:800]}"
        )

    candidate = cleaned_text[start_index:]

    open_curly = candidate.count("{")
    close_curly = candidate.count("}")
    open_square = candidate.count("[")
    close_square = candidate.count("]")

    if close_square < open_square:
        candidate += "]" * (open_square - close_square)
    if close_curly < open_curly:
        candidate += "}" * (open_curly - close_curly)

    decoder = json.JSONDecoder()

    for idx, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed_json, _ = decoder.raw_decode(
                candidate[idx:]
            )
            return parsed_json
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"No JSON found.\n\n{candidate[:800]}"
    )


def infer_problem_type(
        df,
        target_column
):

    y = df[
        target_column
    ]

    if str(
        y.dtype
    ) == "object":

        return "classification"

    if y.nunique() <= 20:

        return "classification"

    return "regression"


def infer_problem_type_from_metadata(
        metadata,
        target_column
):

    dtypes = metadata.get(
        "dtypes",
        {}
    )

    nunique = metadata.get(
        "nunique",
        {}
    )

    rows = int(
        metadata.get(
            "rows",
            0
        )
    )

    target_dtype = str(
        dtypes.get(
            target_column,
            ""
        )
    ).lower()

    target_unique = int(
        nunique.get(
            target_column,
            0
        )
    )

    if rows <= 0:
        return "classification"

    unique_ratio = target_unique / rows
    low_cardinality_cutoff = min(
        20,
        max(
            2,
            int(rows * 0.05)
        )
    )

    if (
        "object" in target_dtype
        or
        "category" in target_dtype
        or
        "bool" in target_dtype
    ):
        return "classification"

    if "int" in target_dtype:
        if target_unique <= low_cardinality_cutoff:
            return "classification"
        return "regression"

    if "float" in target_dtype:
        if target_unique <= low_cardinality_cutoff:
            return "classification"
        return "regression"

    return "regression"


def build_target_summary(
        metadata,
        target_column
):

    dtypes = metadata.get(
        "dtypes",
        {}
    )

    nunique = metadata.get(
        "nunique",
        {}
    )

    missing_values = metadata.get(
        "missing_values",
        {}
    )

    sample_rows = metadata.get(
        "sample_rows",
        {}
    )

    summary_statistics = metadata.get(
        "summary_statistics",
        {}
    )

    return {

        "target_column":
        target_column,

        "dtype":
        dtypes.get(
            target_column,
            ""
        ),

        "unique_values":
        nunique.get(
            target_column,
            0
        ),

        "missing_values":
        missing_values.get(
            target_column,
            0
        ),

        "sample_values":
        list(
            sample_rows.get(
                target_column,
                {}
            ).values()
        ),

        "summary_statistics":
        summary_statistics.get(
            target_column,
            {}
        )
    }


def analyze_dataset(
        metadata,
        description,
        target_column
):

    target_summary = build_target_summary(
        metadata,
        target_column
    )

    prompt = f"""
You are an expert Machine Learning Engineer.

Dataset Metadata:
{metadata}

Target Column Name:
{target_column}

Target Column Evidence:
{target_summary}

Dataset Description:
{description}

Important instructions:

- Read the target column evidence first.
- Decide the problem type primarily from the target column dtype, unique values, sample values, and summary statistics.
- Do NOT choose classification just because the description sounds like a classification task.
- If the target column is numeric with many distinct values, prefer regression.
- The user has already selected the target column.

DO NOT identify another target column.

Your tasks:

1. Identify problem type
   - classification
   - regression

2. Identify numerical features

3. Identify categorical features

4. Recommend:
   - missing value strategy
   - encoding strategy
   - scaling strategy

5. Suggest SPECIFIC feature engineering ideas.

IMPORTANT:

Do NOT return generic operations.

BAD:

{{
    "operation":"ratio"
}}

GOOD:

{{
    "operation":"ratio",
    "columns":[
        "salary",
        "age"
    ],
    "new_feature_name":
    "salary_age_ratio"
}}

Supported operations:

- ratio
- interaction
- sum
- difference
- log_transform

Return ONLY valid JSON.

Example:

{{
    "problem_type":"classification",

    "numerical_features":[
        "Age",
        "Salary"
    ],

    "categorical_features":[
        "City"
    ],

    "missing_value_strategy":"median",

    "encoding_strategy":"one_hot",

    "scaling_strategy":"standard",

    "feature_engineering":[

        {{
            "operation":"ratio",
            "columns":[
                "Salary",
                "Age"
            ],
            "new_feature_name":
            "salary_age_ratio"
        }}

    ]
}}
"""

    response = (
        client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    )

    result = (
        response
        .choices[0]
        .message
        .content
    )

    result_json = (
        extract_json(
            result
        )
    )

    if not isinstance(
        result_json,
        dict
    ):
        result_json = {}

    llm_problem_type = result_json.get(
        "problem_type"
    )

    heuristic_problem_type = (
        infer_problem_type_from_metadata(
            metadata,
            target_column
        )
    )

    if llm_problem_type not in [
        "classification",
        "regression"
    ]:
        result_json["problem_type"] = (
            heuristic_problem_type
        )

    elif (
        llm_problem_type == "regression"
        and
        heuristic_problem_type == "classification"
    ):
        result_json["problem_type"] = (
            "classification"
        )

    result_json[
        "target_column"
    ] = target_column

    return result_json
