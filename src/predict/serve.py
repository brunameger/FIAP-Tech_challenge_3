# src/predict/serve.py
import os
import joblib
import tempfile
from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd

app = Flask(__name__)

GCS_BUCKET = os.environ.get("GCS_BUCKET", "fiap-tech3-models")
MODEL_NAME = os.environ.get("MODEL_NAME", "ibov_xgb_v1.joblib")

def download_model(bucket_name, model_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_name)
    tmp = f"/tmp/{model_name}"
    blob.download_to_filename(tmp)
    data = joblib.load(tmp)
    return data

model_data = download_model(GCS_BUCKET, MODEL_NAME)
model = model_data["model"]
features = model_data["features"]

def preprocess_input(df):
    # expects theoricalQty, cod, asset and optionally data_referencia
    df = df.copy()
    # simple features similar to train (no lags here); you can enhance to fetch last historic lags
    df["dow"] = pd.to_datetime(df.get("data_referencia", pd.Timestamp.now())).dt.dayofweek
    df["month"] = pd.to_datetime(df.get("data_referencia", pd.Timestamp.now())).dt.month
    df["cod_cat"] = df["cod"].astype("category").cat.codes
    # keep features order
    return df[features]

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
    X = preprocess_input(df)
    preds = model.predict(X)
    df["predicted_part"] = preds
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
