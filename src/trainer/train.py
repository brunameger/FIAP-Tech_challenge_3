# src/trainer/train.py
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from google.cloud import storage
from src.utils.bq_utils import load_ibov_table

# Config
BQ_TABLE = os.environ.get("BQ_TABLE", "fiap-tech3.tc_dataset.ibov")
MODEL_NAME = os.environ.get("MODEL_NAME", "ibov_xgb_v1.joblib")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "fiap-tech3-models")

# ---------------- Feature engineering ----------------
def create_features(df):
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    df = df.sort_values(["cod", "data_referencia"])

    # Lags e rolling
    df["theor_lag1"] = df.groupby("cod")["theoricalQty"].shift(1)
    df["theor_lag2"] = df.groupby("cod")["theoricalQty"].shift(2)
    df["roll_mean_3"] = df.groupby("cod")["theoricalQty"].shift(1).rolling(3, min_periods=1).mean().reset_index(0, drop=True)

    # Features de data
    df["dow"] = df["data_referencia"].dt.dayofweek
    df["month"] = df["data_referencia"].dt.month
    df["cod_cat"] = df["cod"].astype("category").cat.codes

    # Preencher lags e rolling com valor anterior se houver NaN
    df[["theor_lag1", "theor_lag2", "roll_mean_3"]] = df.groupby("cod")[["theor_lag1", "theor_lag2", "roll_mean_3"]].fillna(method="ffill")
    
    # Último recurso: se ainda tiver NaN, preencher com valor base
    df[["theor_lag1", "theor_lag2", "roll_mean_3"]] = df[["theor_lag1", "theor_lag2", "roll_mean_3"]].fillna(df["theoricalQty"])

    return df

# ---------------- Treino e avaliação ----------------
def train_and_evaluate(df):
    df = create_features(df)
    FEATURES = ["theor_lag1", "theor_lag2", "roll_mean_3", "dow", "month", "cod_cat"]
    X = df[FEATURES]
    y = df["theoricalQty"]

    tscv = TimeSeriesSplit(n_splits=5)
    rmses, models = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse",
            "seed": 42
        }

        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=300,
                          evals=[(dtrain,"train"),(dtest,"eval")],
                          early_stopping_rounds=20, verbose_eval=False)

        rmses.append(np.sqrt(np.mean((y_test - model.predict(dtest))**2)))
        models.append(model)

    best_idx = int(np.argmin(rmses))
    best_model = models[best_idx]
    return best_model, FEATURES

# ---------------- Upload para GCS ----------------
def upload_to_gcs(local_path, bucket_name, dest_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Modelo enviado: gs://{bucket_name}/{dest_blob_name}")

# ---------------- Main ----------------
def main():
    df = load_ibov_table(BQ_TABLE)
    model, feat_names = train_and_evaluate(df)

    import tempfile
    local_path = os.path.join(tempfile.gettempdir(), MODEL_NAME)
    joblib.dump({"model": model, "features": feat_names}, local_path)
    upload_to_gcs(local_path, GCS_BUCKET, MODEL_NAME)
    print("✅ Modelo treinado e enviado com sucesso!")

if __name__ == "__main__":
    main()
