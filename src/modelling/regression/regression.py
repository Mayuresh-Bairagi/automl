import os
import uuid
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb


# --- Safe MAPE implementation ---
def safe_mape(y_true, y_pred, epsilon=1e-10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))


class AutoMLRegressor:
    def __init__(self, save_dir: str = None, scoring: str = "r2"):
        # Default save directory
        if save_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(base_dir, "RegressionModels")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.scoring = scoring
        self.results = []
        self.best_model_info = None
        self.best_model_obj = None

    def _evaluate_model(self, model, X_test, y_test, params):
        """Evaluate a trained model"""
        y_pred = model.predict(X_test)
        return {
            "model": model.named_steps["model"].__class__.__name__,
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "mape": safe_mape(y_test, y_pred),
            "best_params": params,
            "estimator": model
        }

    def _run_search(self, model, param_grid, X_train, y_train, X_test, y_test):
        """Run GridSearch and evaluate"""
        grid = GridSearchCV(model, param_grid, cv=3, scoring=self.scoring, n_jobs=-1)
        grid.fit(X_train, y_train)
        return self._evaluate_model(grid.best_estimator_, X_test, y_test, grid.best_params_)

    def fit(self, df: pd.DataFrame, target: str):
        """Train multiple regression models and select the best one"""
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Models & grids ---
        model_candidates = [
            (
                Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
                {}
            ),
            (
                Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor(random_state=42))]),
                {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}
            ),
            (
                Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))]),
                {"model__n_estimators": [50, 100], "model__max_depth": [None, 10, 20]}
            ),
            (
                Pipeline([("scaler", StandardScaler()), ("model", xgb.XGBRegressor(random_state=42, verbosity=0))]),
                {"model__n_estimators": [50, 100], "model__max_depth": [3, 5, 7], "model__learning_rate": [0.01, 0.1, 0.2]}
            ),
            (
                Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
                {"model__n_neighbors": [3, 5, 7, 10]}
            ),
            (
                Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(max_iter=1000, random_state=42))]),
                {"model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
                 "model__activation": ["relu", "tanh"],
                 "model__solver": ["adam"]}
            )
        ]

        # --- Train & evaluate all ---
        self.results = []
        for model, grid in model_candidates:
            if grid:
                res = self._run_search(model, grid, X_train, y_train, X_test, y_test)
            else:
                model.fit(X_train, y_train)
                res = self._evaluate_model(model, X_test, y_test, params="Default")
            self.results.append(res)

        # --- Select best model ---
        self.best_model_info = max(self.results, key=lambda x: x["r2"])
        self.best_model_obj = self.best_model_info["estimator"]

        # --- Save model ---
        model_id = str(uuid.uuid4())[:8]
        model_name = self.best_model_info["model"]
        model_path = os.path.join(self.save_dir, f"{model_name}_{model_id}.joblib")
        joblib.dump(self.best_model_obj, model_path)

        # Save metadata
        metadata = {
            "model": model_name,
            "best_params": self.best_model_info["best_params"],
            "metrics": {
                "r2": self.best_model_info["r2"],
                "mae": self.best_model_info["mae"],
                "rmse": self.best_model_info["rmse"],
                "mape": self.best_model_info["mape"],
            },
            "download_path": model_path,
            "model_id": model_id
        }
        json_path = os.path.join(self.save_dir, f"{model_name}_{model_id}.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata, self.results

    def predict(self, X: pd.DataFrame):
        if self.best_model_obj is None:
            raise ValueError("No model trained yet. Run fit() first.")
        return self.best_model_obj.predict(X)

    @staticmethod
    def load_model(path: str):
        return joblib.load(path)
