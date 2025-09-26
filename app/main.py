from fastapi import FastAPI, UploadFile, File,HTTPException
from pathlib import Path
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from src.dataCleaning.featureEngineering01 import FeatureEngineer1
from src.data_dashboard.eda import EDA
from model.models import requestEDA
from fastapi.staticfiles import StaticFiles

app = FastAPI()

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
folder_path=  r'data\datasetAnalysis'
app.mount("/data", StaticFiles(directory=folder_path), name="data")

@app.get("/")
async def root():
    return {"message": "Welcome to the AutoML API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        df = None


        if file.filename.endswith(".csv"): 
            df = pd.read_csv(BytesIO(contents)) 
        elif file.filename.endswith((".xlsx", ".xls")): 
            df = pd.read_excel(BytesIO(contents)) 
        else: 
            return {"error": "Unsupported file type"}
        fe = FeatureEngineer1(df)
        df, session_id = fe.generate_features()

        preview = df.head(10).to_dict(orient="records")
        print(df.shape)
        return {"filename": file.filename, "preview": preview,"session_id" : session_id}

    except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))


@app.post("/eda")
async def eda(request:requestEDA):
    try:
        session_id = request.session_id
        eda_obj = EDA(session_id=session_id)
        html_path = eda_obj.generate_report()
        html_url = f"http://127.0.0.1:8000/data/{session_id}/index.html"

        print(f"EDA report URL: {html_url}")
        return {"session_id": session_id, "eda_html_path": html_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
