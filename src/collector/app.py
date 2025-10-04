# src/app.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PIPE_PATH = "artifacts/pipeline_ibov.joblib"
app = FastAPI(title="IBOV Part Predict API")

# carregar pipeline
pipeline = joblib.load(MODEL_PIPE_PATH)

class PredictRequest(BaseModel):
    cod: str
    asset: str
    type: str
    theoricalQty: float = None
    ref_year: int = None
    ref_month: int = None

@app.post("/prever")
def prever(req: PredictRequest):
    df = pd.DataFrame([req.dict()])
    # se faltar ref_year/ref_month, inserir ano/mes de hoje
    if df["ref_year"].isnull().all():
        df["ref_year"] = pd.Timestamp.now().year
    if df["ref_month"].isnull().all():
        df["ref_month"] = pd.Timestamp.now().month

    try:
        pred = pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"previsao_part": float(pred[0])}
