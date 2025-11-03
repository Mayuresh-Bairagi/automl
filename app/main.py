from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from io import BytesIO
import pandas as pd
import uvicorn

from src.dataCleaning.featureEngineering01 import FeatureEngineer1
from src.data_dashboard.eda import EDA
from src.Regression.regression import AutoMLRegressor
from model.models import requestEDA, request_ml_models
from src.problem_statement.target_variable import TargetVariable
from src.Classifier.MLClassifier import AutoMLClassifier


app = FastAPI(title="AutoML Backend", description="FastAPI backend for AutoML project")

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

folder_path = Path("data/datasetAnalysis")
folder_path.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=folder_path), name="data")



@app.get("/")
async def root():
    return {"message": "Welcome to the AutoML API"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel.")

        fe = FeatureEngineer1(df)
        df, session_id = fe.generate_features()

        preview = df.head(10).to_dict(orient="records")
        print(f"[UPLOAD] File: {file.filename}, Shape: {df.shape}, Session: {session_id}")

        return {
            "filename": file.filename,
            "preview": preview,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/eda")
async def eda(request: requestEDA):
    try:
        session_id = request.session_id
        eda_obj = EDA(session_id=session_id)
        html_path = eda_obj.generate_report()

        html_url = f"http://127.0.0.1:8000/data/{session_id}/index.html"
        print(f"[EDA] Report generated: {html_url}")

        return {"session_id": session_id, "eda_html_path": html_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")


@app.post("/ml-models")
async def ml_model(request: request_ml_models):
    try:
        session_id = request.session_id
        problem_statement = request.problem_statement
        target_var_handler = TargetVariable(session_id=session_id)
        result, df = target_var_handler.get_target_variable(problem_statement)

        problem_statement_type = result['problem_type']

        if problem_statement_type.lower() == "regression":
            automl_regressor = AutoMLRegressor(
                session_id=session_id,
                problem_statement=problem_statement,
                result=result,
                df=df
            )

            results_df, trained_models, model_paths = automl_regressor.train_models(skip_heavy=True)

            results_dict = results_df.to_dict(orient="records")

            

            print(f"[ML MODEL] Training completed. Models: {len(model_paths)}")

            return {
                "session_id": session_id,
                "results": results_dict,
                "model_paths": model_paths
            }
        
        elif problem_statement_type.lower() == "classification":
            automl_classifier = AutoMLClassifier(
                session_id=session_id,
                problem_statement=problem_statement,
                result=result,
                df=df
            )

            results_df, trained_models, model_paths = automl_classifier.train_models(skip_heavy=True)

            results_dict = results_df.to_dict(orient="records")

            print(f"[ML MODEL] Training completed. Models: {len(model_paths)}")

            return {
                "session_id": session_id,
                "results": results_dict,
                "model_paths": model_paths
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported problem type: {problem_statement_type}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
