import json
import re

from utils.llm import (
    client,
    DEFAULT_MODEL
)


def extract_json(response_text):

    response_text = re.sub(
        r"<think>.*?</think>",
        "",
        response_text,
        flags=re.DOTALL
    )

    response_text = response_text.replace(
        "```json",
        ""
    )

    response_text = response_text.replace(
        "```",
        ""
    )

    match = re.search(
        r"\{.*\}",
        response_text,
        re.DOTALL
    )

    if not match:

        raise ValueError(
            f"No JSON found.\n\n{response_text}"
        )

    return json.loads(
        match.group(0)
    )


def build_column_summary(
        df,
        target_column
):

    summary = {}

    for col in df.columns:

        if col == target_column:
            continue

        summary[col] = {

            "dtype":
            str(
                df[col].dtype
            ),

            "missing_percentage":
            round(
                (
                    df[col]
                    .isnull()
                    .mean()
                ) * 100,
                2
            ),

            "unique_values":
            int(
                df[col]
                .nunique(
                    dropna=True
                )
            ),

            "unique_ratio":
            round(
                (
                    df[col]
                    .nunique(
                        dropna=True
                    )
                    /
                    max(
                        len(df),
                        1
                    )
                ),
                4
            )
        }

    return summary


def analyze_features_with_llm(
        df,
        target_column,
        dataset_description=""
):

    column_summary = (
        build_column_summary(
            df,
            target_column
        )
    )

    prompt = f"""
You are an expert Machine Learning Engineer.

Dataset Description:
{dataset_description}

Target Column:
{target_column}

Column Statistics:
{column_summary}

Your job:

1. Identify columns that should be dropped.
2. Identify possible leakage columns.
3. Identify columns that should be transformed.
4. Identify columns that should definitely be retained.

Consider:

- Identifier columns
- Useless columns
- High cardinality columns
- High missing columns
- Potential target leakage
- Date columns
- Free-text columns

Return ONLY JSON.

Example:

{{
    "drop_columns":[
        {{
            "column":"CustomerID",
            "reason":"Identifier column"
        }}
    ],

    "potential_leakage":[
        {{
            "column":"Default_Status",
            "reason":"Contains target information"
        }}
    ],

    "transform_columns":[
        {{
            "column":"OrderDate",
            "action":"extract_year_month_day"
        }}
    ],

    "keep_columns":[
        "Age",
        "Salary"
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

    print(
        "\n===== FEATURE CLEANER AGENT =====\n"
    )

    print(result)

    print(
        "\n=================================\n"
    )

    return extract_json(
        result
    )