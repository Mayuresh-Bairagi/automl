from Propmt.propmt_lib import PROMPT_REGISTRY
from langchain.output_parsers import PydanticOutputParser
from model.models import FeatureSelectionOutput
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from logger.customlogger import CustomLogger
from expection.customExpection import AutoML_Exception
from src.problem_statement.target_variable import TargetVariable

class FeatureSelector:
    def __init__(self, session_id, problem_statement):
        self.logger = CustomLogger().get_logger(__name__)
        self.problem_statement = problem_statement  

        try:
            self.logger.info("Initializing TargetVariable handler", extra={'session_id': session_id})
            self.target_var_handler = TargetVariable(session_id=session_id)
            self.result, self.df = self.target_var_handler.get_target_variable(problem_statement)

            self.target = self.result['target_variable']
            self.problem_type = self.result['problem_type']

            self.feature_selection_prompt = PROMPT_REGISTRY['feature_selection']
            self.parser = JsonOutputParser(pydantic_object=FeatureSelectionOutput)

            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            # Build the chain
            self.chain = self.feature_selection_prompt | self.llm | self.fixing_parser

            self.logger.info("FeatureSelector initialized successfully", 
                             extra={'target': self.target, 'problem_type': self.problem_type})

        except Exception as e:
            self.logger.error("Initialization failed", exc_info=True)
            raise AutoML_Exception("Failed to initialize FeatureSelector", e)

    def select_features(self):
        try:
            self.logger.info("Selecting features", extra={'problem_type': self.problem_type})

            if self.problem_type == "regression":
                return self._regression_selection()
            elif self.problem_type == "classification":
                return self._classification_selection()
            elif self.problem_type == "clustering":
                return self._clustering_selection()
            else:
                raise ValueError(f"Unknown problem type: {self.problem_type}")

        except Exception as e:
            self.logger.error("Feature selection failed", exc_info=True)
            raise AutoML_Exception("Feature selection process failed", e)

    def _regression_selection(self):
        try:
            if not np.issubdtype(self.df[self.target].dtype, np.number):
                raise ValueError("Regression target must be numeric")

            corr = self.df.corr(numeric_only=True)
            if self.target not in corr.columns:
                raise ValueError("Target column not numeric or not in DataFrame")

            target_corr = corr[self.target].sort_values(ascending=False)
            selected_features = target_corr.drop(labels=self.target).index.tolist()

            self.logger.info("Regression feature selection complete", 
                             extra={'selected_count': len(selected_features)})

            return {
                "features": selected_features,
                "meta": {"method": "correlation", "target_corr": target_corr.to_dict()},
                "dtypes": self.df.dtypes.astype(str).to_dict(),
                "correlation_matrix": corr.to_dict()
            }
        except Exception as e:
            self.logger.error("Regression selection failed", exc_info=True)
            raise AutoML_Exception("Regression feature selection failed", e)

    def _classification_selection(self):
        try:
            X = self.df.drop(columns=[self.target])
            y = self.df[self.target]

            # Ensure non-negative for chi2
            X_nonneg = X.copy()
            for col in X_nonneg.select_dtypes(include=[np.number]).columns:
                min_val = X_nonneg[col].min()
                if min_val < 0:
                    X_nonneg[col] = X_nonneg[col] - min_val

            chi_scores, _ = chi2(X_nonneg.select_dtypes(include=[np.number]), y)
            mi_scores = mutual_info_classif(X, y, discrete_features='auto')

            results = pd.DataFrame({
                "feature": X.columns,
                "chi2_score": np.pad(chi_scores, (0, max(0, len(X.columns)-len(chi_scores))), 'constant'),
                "mutual_info": mi_scores
            }).sort_values("mutual_info", ascending=False)

            selected_features = results["feature"].tolist()

            self.logger.info("Classification feature selection complete", 
                             extra={'selected_count': len(selected_features)})

            return {
                "features": selected_features,
                "meta": results.to_dict(orient="list"),
                "dtypes": self.df.dtypes.astype(str).to_dict(),
                "correlation_matrix": self.df.corr(numeric_only=True).to_dict()
            }
        except Exception as e:
            self.logger.error("Classification selection failed", exc_info=True)
            raise AutoML_Exception("Classification feature selection failed", e)

    def _clustering_selection(self):
        try:
            X = self.df
            corr = X.corr(numeric_only=True)

            selector = VarianceThreshold(threshold=0.01)
            X_var = selector.fit_transform(X)
            kept_features = X.columns[selector.get_support()]

            pca = PCA(n_components=min(5, X_var.shape[1]))
            X_pca = pca.fit_transform(X_var)

            self.logger.info("Clustering feature selection complete", 
                             extra={'kept_features': list(kept_features), 'pca_components': pca.n_components_})

            return {
                "features": list(kept_features),
                "meta": {
                    "method": "correlation + variance + PCA",
                    "pca_explained_variance": pca.explained_variance_ratio_.tolist()
                },
                "dtypes": self.df.dtypes.astype(str).to_dict(),
                "correlation_matrix": corr.to_dict(),
                "sample_data": self.df.head().to_dict(orient="records")
            }
        except Exception as e:
            self.logger.error("Clustering selection failed", exc_info=True)
            raise AutoML_Exception("Clustering feature selection failed", e)

    def llm_response(self):
        try:
            self.logger.info("Generating LLM response for feature selection")
            meta_data = self.select_features()
            meta_data['problem_statement'] = self.problem_statement
            meta_data['target_col'] = self.target

            response = self.chain.invoke({
                "return_instructions": self.parser.get_format_instructions(),
                "metadata_json": meta_data
            })

            self.logger.info("LLM response generated successfully")
            return response
        except Exception as e:
            self.logger.error("LLM response generation failed", exc_info=True)
            raise AutoML_Exception("LLM response generation failed", e)


if __name__ == "__main__":
    session_id = "session_id_20250925_101824_7a7c48d7"
    problem_statement = "Predict the price of plane ticket"

    try:
        selector = FeatureSelector(session_id, problem_statement)
        context = selector.llm_response()
        print(context)

    except Exception as e:
        print(e)
