from fastapi import FastAPI
import pandas as pd
import joblib
import xgboost as xgb
from datetime import timedelta
from src.utils.bq_utils import load_ibov_table

app = FastAPI()

MODEL_PATH = "ibov_xgb_v1.joblib"

@app.post("/predict")
def predict_next_day():
    # Carrega o modelo
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"] if isinstance(model_data, dict) else model_data

    # Carrega dados do BigQuery
    df = load_ibov_table("fiap-tech3.tc_dataset.ibov")

    if df.empty:
        return {"error": "Sem dados para prever"}

    # Ajusta colunas e ordena
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values(["cod", "data_referencia"])

    # Previsão do próximo dia
    results = []
    for cod in df["cod"].unique():
        df_ativo = df[df["cod"] == cod].copy()
        df_ativo = df_ativo.sort_values("data_referencia")

        last_row = df_ativo.iloc[-1]
        next_date = last_row["data_referencia"] + timedelta(days=1)

        theor_lag1 = last_row["theoricalQty"]
        theor_lag2 = df_ativo["theoricalQty"].iloc[-2] if len(df_ativo) > 1 else theor_lag1
        roll_mean_3 = df_ativo["theoricalQty"].iloc[-3:].mean() if len(df_ativo) >= 3 else df_ativo["theoricalQty"].mean()
        dow = next_date.dayofweek
        month = next_date.month
        cod_cat_series = df_ativo["cod"].astype("category").cat.codes
        cod_cat = cod_cat_series.iloc[-1]  # Pega o código do último registro

        features = {
            "theor_lag1": theor_lag1,
            "theor_lag2": theor_lag2,
            "roll_mean_3": roll_mean_3,
            "dow": dow,
            "month": month,
            "cod_cat": cod_cat
        }

        X_pred = pd.DataFrame([features])
        dtest = xgb.DMatrix(X_pred)
        y_pred = model.predict(dtest)[0]

        results.append({
            "cod": cod,
            "asset": last_row["asset"],
            "data_referencia": next_date.strftime("%Y-%m-%d"),
            "prediction": float(y_pred)
        })

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
