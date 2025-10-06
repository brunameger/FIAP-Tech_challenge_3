# src/predict/serve.py
from flask import Flask, jsonify
import pandas as pd
import xgboost as xgb
import joblib, os
from google.cloud import bigquery, storage

app = Flask(__name__)

MODEL_PATH = "ibov_xgb_v1.joblib"
GCS_BUCKET = "fiap-tech3-models"
BQ_CLIENT = bigquery.Client()
DATASET = "tc_dataset"
TABLE = "ibov"

# Baixar modelo se não existir
if not os.path.exists(MODEL_PATH):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    bucket.blob(MODEL_PATH).download_to_filename(MODEL_PATH)

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

def add_features(df):
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values(["cod", "data_referencia"])
    df["theor_lag1"] = df.groupby("cod")["theoricalQty"].shift(1)
    df["theor_lag2"] = df.groupby("cod")["theoricalQty"].shift(2)
    df["roll_mean_3"] = df.groupby("cod")["theoricalQty"].shift(1).rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df["dow"] = df["data_referencia"].dt.dayofweek
    df["month"] = df["data_referencia"].dt.month
    df["cod_cat"] = df["cod"].astype("category").cat.codes

    # Preencher NaN
    df[["theor_lag1","theor_lag2","roll_mean_3"]] = df.groupby("cod")[["theor_lag1","theor_lag2","roll_mean_3"]].fillna(method="ffill")
    df[["theor_lag1","theor_lag2","roll_mean_3"]] = df[["theor_lag1","theor_lag2","roll_mean_3"]].fillna(df["theoricalQty"])
    return df

def preprocess_input(df, feature_list):
    return df[feature_list]

@app.route("/predict", methods=["POST"])
def predict():
    query = f"""
    SELECT cod, asset, theoricalQty, data_referencia
    FROM `{BQ_CLIENT.project}.{DATASET}.{TABLE}`
    ORDER BY cod, data_referencia
    """
    df = BQ_CLIENT.query(query).to_dataframe()
    if df.empty:
        return jsonify({"error":"Sem dados"}),400

    df = add_features(df)
    X = preprocess_input(df, features)
    dmatrix = xgb.DMatrix(X)
    df["prediction"] = model.predict(dmatrix)

    return jsonify(df[["data_referencia","cod","asset","prediction"]].to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
