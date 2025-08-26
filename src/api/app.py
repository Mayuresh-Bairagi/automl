from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import pandas as pd
import uvicorn
import os
from ..modelling.regression.regression import AutoMLRegressor

app = FastAPI()

@app.post("/train/")
async def train_model(file: UploadFile, target: str = Form(...)):
    """
    Upload a CSV file and specify the target column for training.
    """
    # Read CSV into pandas DataFrame
    df = pd.read_csv(file.file)

    # Initialize AutoMLRegressor
    automl = AutoMLRegressor()

    # Train the model
    meta, results = automl.fit(df, target)

    # Clean results (remove estimator objects)
    cleaned_results = []
    for r in results:
        cleaned_results.append({
            "model": r["model"],
            "r2": r["r2"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "mape": r["mape"],
            "best_params": r["best_params"]
        })

    # Build a download link (relative path inside project)
    download_filename = os.path.basename(meta["download_path"])
    download_url = f"/download/{download_filename}"

    return {
        "status": "success",
        "best_model": meta,
        "all_models": cleaned_results,
        "download_link": download_url
    }

@app.get("/download/{filename}")
async def download_model(filename: str):
    """
    Download a trained model file by filename.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "..", "modelling", "regression", "RegressionModels")
    file_path = os.path.join(model_dir, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    return FileResponse(file_path, filename=filename)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
