import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception
import joblib
from src.problem_statement.AutoFeatureSelector import FeatureSelector

class AutoMLRegressor:
    def __init__(self, session_id, problem_statement, test_size: float = 0.2, random_state: int = 42, df=None, result=None):
        self.logger = CustomLogger().get_logger(__file__)
        self.logger.info("Initializing AutoMLRegressor", session_id=session_id)

        try:
            self.database_path = os.path.join(os.getcwd(), 'data', 'datasetAnalysis', session_id, 'processed_file.csv')
            if not os.path.exists(self.database_path):
                raise FileNotFoundError(f"Dataset not found at {self.database_path}")
            self.df = pd.read_csv(self.database_path)
            self.selector = FeatureSelector(session_id, problem_statement,result, df)
            self.context = self.selector.llm_response()

            self.target_col = self.context['target_col']
            self.dropped_features = self.context['dropped_features'] or []
            self.test_size = test_size
            self.random_state = random_state
            self.output_dir = os.path.join(os.getcwd(), 'data', 'datasetAnalysis', session_id)
            os.makedirs(self.output_dir, exist_ok=True)

            self.label_encoders = {}
            self.scaler = None
            self.best_model = None
            self.results_df = None
            self.trained_models = {}
            self.model_paths = {}
            self.preprocessing_objects = {}

            self.logger.info("AutoMLRegressor initialized successfully")

        except Exception as e:
            self.logger.error("Initialization failed", error=str(e))
            raise AutoML_Exception("Initialization failed", e)

    def preprocess(self):
        try:
            self.logger.info("Starting preprocessing")

            df_cleaned = self.df.drop(columns=self.dropped_features, errors='ignore')
            X = df_cleaned.drop(self.target_col, axis=1)
            y = df_cleaned[self.target_col]

            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le

            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            self.scaler = MinMaxScaler()
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

            self.preprocessing_objects = {
                "scaler": self.scaler,
                "label_encoders": self.label_encoders
            }
            joblib.dump(self.preprocessing_objects, os.path.join(self.output_dir, "preprocessing.joblib"))

            self.logger.info("Preprocessing completed", numeric_features=list(numeric_cols), categorical_features=list(self.label_encoders.keys()))
            return X, y

        except Exception as e:
            self.logger.error("Preprocessing failed", error=str(e))
            raise AutoML_Exception("Preprocessing failed", e)

    def train_models(self, cv=3, skip_heavy=False):
        self.logger.info("Starting preprocessing inside train_models()")
        X, y = self.preprocess()
        models = {
            'LinearRegression': (LinearRegression(), {'fit_intercept': [True, False]}),
            'Ridge': (Ridge(), {'alpha': [0.1, 1, 10]}),
            'Lasso': (Lasso(max_iter=3000), {'alpha': [0.01, 0.1, 1]}),
            'ElasticNet': (ElasticNet(max_iter=3000), {'alpha': [0.01, 0.1, 1], 'l1_ratio': [0.5, 0.7]}),
            'RandomForest': (RandomForestRegressor(random_state=self.random_state), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
            'GradientBoosting': (GradientBoostingRegressor(random_state=self.random_state), {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}),
            'SVR': (SVR(), {'kernel': ['linear', 'rbf'], 'C': [0.1, 1]})
        }

        if skip_heavy:
            models = {k: v for k, v in models.items() if k not in ['RandomForest', 'GradientBoosting']}

        try:
            self.logger.info("Splitting dataset", test_size=self.test_size)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        except Exception as e:
            self.logger.error("Dataset split failed", error=str(e))
            raise AutoML_Exception("Dataset split failed", e)

        results = []
        best_score = -float('inf')

        for name, (model, params) in models.items():
            try:
                self.logger.info(f"Training {name}")
                gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)
                y_pred = gs.predict(X_test)
                score = r2_score(y_test, y_pred)

                results.append({
                    'Model': name,
                    'R2_Score': score,
                    'Best_Params': gs.best_params_
                })

                model_path = os.path.join(self.output_dir, f"{name}.joblib")
                joblib.dump(gs.best_estimator_, model_path)
                self.trained_models[name] = gs.best_estimator_
                self.model_paths[name] = model_path

                if score > best_score:
                    best_score = score
                    self.best_model = gs.best_estimator_

                self.logger.info(f"{name} trained", R2_Score=score, model_path=model_path)

            except Exception as e:
                self.logger.error(f"{name} training failed", error=str(e))
                continue

        self.results_df = pd.DataFrame(results).sort_values(by='R2_Score', ascending=False)
        self.logger.info("All models training completed", trained_models=list(self.trained_models.keys()))
        return self.results_df, self.trained_models, self.model_paths





if __name__ ==  "__main__":
    try:
        session_id = "session_id_20251004_193459_4af2e06b"
        problem_statement = "Predict the ticket price"

        automl = AutoMLRegressor(session_id, problem_statement)
        results_df, trained_models, model_paths = automl.train_models(skip_heavy=True)

        print("Model Training Summary")
        print(results_df)
        print("\nBest Model:", results_df.iloc[0]['Model'])
        print("Model Saved at:", model_paths[results_df.iloc[0]['Model']])

    except Exception as e:
        print("Error during AutoML execution:", str(e))