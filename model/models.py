from pydantic import BaseModel , RootModel
from typing import List , Union, Literal
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