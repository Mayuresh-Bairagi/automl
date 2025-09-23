from fastapi import FastAPI, UploadFile, File
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from src.dataCleaning.featureEngineering01 import FeatureEngineer1

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
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
