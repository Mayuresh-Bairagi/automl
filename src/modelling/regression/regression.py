import os
import uuid
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Import custom exception and logger
from expection.customExpection import AutoML_Exception
from logger.customlogger import CustomLogger


# --- Safe MAPE implementation ---
def safe_mape(y_true, y_pred, epsilon=1e-10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))


class AutoMLRegressor:
    def __init__(self, save_dir: str = None, scoring: str = "r2"):
        try:
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

            # Logger
            self.logger = CustomLogger().get_logger(__file__)
            self.logger.info("AutoMLRegressor initialized", save_dir=self.save_dir)

        except Exception as e:
            raise AutoML_Exception(e)

    def _evaluate_model(self, model, X_test, y_test, params):
        """Evaluate a trained model"""
        try:
            y_pred = model.predict(X_test)
            result = {
                "model": model.named_steps["model"].__class__.__name__,
                "r2": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": mean_squared_error(y_test, y_pred, squared=False),
                "mape": safe_mape(y_test, y_pred),
                "best_params": params,
                "estimator": model
            }
            self.logger.info("Model evaluated", model=result["model"], r2=result["r2"])
            return result
        except Exception as e:
            self.logger.error("Error in _evaluate_model", error=str(e))
            raise AutoML_Exception(e)

    def _run_search(self, model, param_grid, X_train, y_train, X_test, y_test):
        """Run GridSearch and evaluate"""
        try:
            grid = GridSearchCV(model, param_grid, cv=3, scoring=self.scoring, n_jobs=-1)
            grid.fit(X_train, y_train)
            self.logger.info("GridSearch completed", model=model.named_steps["model"].__class__.__name__)
            return self._evaluate_model(grid.best_estimator_, X_test, y_test, grid.best_params_)
        except Exception as e:
            self.logger.error("Error in _run_search", error=str(e))
            raise AutoML_Exception(e)

    def fit(self, df: pd.DataFrame, target: str):
        """Train multiple regression models and select the best one"""
        try:
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found in DataFrame")

            X = df.drop(columns=[target])
            y = df[target]

            if X.empty or y.empty:
                raise ValueError("Dataset is empty after preprocessing")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.logger.info("Data split completed", train_size=len(X_train), test_size=len(X_test))

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
                    {"model__n_estimators": [50, 100], "model__max_depth": [3, 5, 7],
                     "model__learning_rate": [0.01, 0.1, 0.2]}
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
                try:
                    if grid:
                        res = self._run_search(model, grid, X_train, y_train, X_test, y_test)
                    else:
                        model.fit(X_train, y_train)
                        res = self._evaluate_model(model, X_test, y_test, params="Default")
                    self.results.append(res)
                except Exception as e:
                    self.logger.error("Model training failed", model=model, error=str(e))
                    continue  # Skip failed model

            if not self.results:
                raise ValueError("No models were successfully trained")

            # --- Select best model ---
            self.best_model_info = max(self.results, key=lambda x: x["r2"])
            self.best_model_obj = self.best_model_info["estimator"]
            self.logger.info("Best model selected", model=self.best_model_info["model"], r2=self.best_model_info["r2"])

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

            self.logger.info("Model training completed", best_model=model_name, model_id=model_id)
            return metadata, self.results

        except Exception as e:
            self.logger.error("Error in fit method", error=str(e))
            raise AutoML_Exception(e)

    def predict(self, X: pd.DataFrame):
        try:
            if self.best_model_obj is None:
                raise ValueError("No model trained yet. Run fit() first.")
            preds = self.best_model_obj.predict(X)
            self.logger.info("Prediction completed", rows=len(X))
            return preds
        except Exception as e:
            self.logger.error("Error in predict", error=str(e))
            raise AutoML_Exception(e)

    @staticmethod
    def load_model(path: str):
        try:
            return joblib.load(path)
        except Exception as e:
            raise AutoML_Exception(e)


if __name__ == "__main__":
    try:
        # Example dummy test
        df = pd.DataFrame({
            "x1": np.random.rand(100),
            "x2": np.random.rand(100),
            "y": np.random.rand(100)
        })
        automl = AutoMLRegressor()
        meta, results = automl.fit(df, "y")
        print(json.dumps(meta, indent=4))
    except Exception as e:
        raise AutoML_Exception(e)
