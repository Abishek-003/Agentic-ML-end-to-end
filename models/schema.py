from pydantic import BaseModel
from typing import List


class FeatureOperation(BaseModel):

    operation: str

    columns: List[str]

    new_feature_name: str


class DatasetAnalysis(BaseModel):

    problem_type: str

    target_column: str

    numerical_features: List[str]

    categorical_features: List[str]

    missing_value_strategy: str

    encoding_strategy: str

    scaling_strategy: str

    feature_engineering: List[FeatureOperation]