from pydantic import BaseModel , RootModel
from typing import List , Union, Literal,Optional
from enum import Enum


class ColumnRecommendation(BaseModel):
    column_name: str
    current_dtype: str
    sample_values : list
    suggested_dtype: Literal["object", "integer", "float", "date", "boolean"]
    reason: str

class DataTypeRecommendation(BaseModel):
    columns: List[ColumnRecommendation]

class FeatureEngineering(BaseModel):
    remarke: Literal["yes", "no"]
    code: str


class requestEDA(BaseModel):
    session_id: str


class TargetVariableRecommendation(BaseModel):
    target_variable: str
    problem_type: Literal["regression", "classification", "clustering"]
    justification: str

class RankedFeature(BaseModel):
    name: str
    score: float
    reason: Optional[str]

class FeatureSelectionOutput(BaseModel):
    target_col : str
    selected_features: List[str]
    dropped_features: List[str]
    ranked_features: List[RankedFeature]


class request_ml_models(BaseModel):
    session_id : str
    problem_statement : str

