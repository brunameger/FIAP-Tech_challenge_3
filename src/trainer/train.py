import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from google.cloud import storage
from src.utils.bq_utils import load_ibov_table
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Config
BQ_TABLE = os.environ.get("BQ_TABLE", "fiap-tech3.tc_dataset.ibov")
MODEL_NAME = os.environ.get("MODEL_NAME", "ibov_xgb_v1.joblib")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "fiap-tech3-models")

def create_features(df):
    # espera colunas: cod, asset, theoricalQty, part, data_referencia
    df = df.sort_values(["cod", "data_referencia"])
    # converter datas
    df["data_referencia"] = pd.to_datetime(df["data_referencia"])
    # lags e rolling por ativo
    df["theor_lag1"] = df.groupby("cod")["theoricalQty"].shift(1)
    df["theor_lag2"] = df.groupby("cod")["theoricalQty"].shift(2)
    df["roll_mean_3"] = df.groupby("cod")["theoricalQty"].shift(1).rolling(3).mean().reset_index(0,drop=True)
    # feature: day of week, month
    df["dow"] = df["data_referencia"].dt.dayofweek
    df["month"] = df["data_referencia"].dt.month
    # encode cod as category codes
    df["cod_cat"] = df["cod"].astype("category").cat.codes
    # drop rows with NA in target or core features
    df = df.dropna(subset=["part","theoricalQty","theor_lag1"])
    return df

def train_and_evaluate(df):
    df = create_features(df)
    FEATURES = ["theoricalQty","theor_lag1","theor_lag2","roll_mean_3","dow","month","cod_cat"]
    X = df[FEATURES]
    y = df["part"].astype(float)

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    rmses = []
    maes = []
    models = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=20,
                  verbose=False)
        ypred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, ypred)
        mae = mean_absolute_error(y_test, ypred)
        rmses.append(rmse)
        maes.append(mae)
        models.append(model)

    print("RMSEs:", rmses, "mean RMSE:", np.mean(rmses))
    print("MAEs:", maes, "mean MAE:", np.mean(maes))
    # escolhe o melhor modelo (menor RMSE)
    best_idx = int(np.argmin(rmses))
    best_model = models[best_idx]
    return best_model, X.columns.tolist()

def upload_to_gcs(local_path, bucket_name, dest_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{dest_blob_name}")

def main():
    df = load_ibov_table(BQ_TABLE)
    model, feat_names = train_and_evaluate(df)
    # salvar local
    local_path = f"/tmp/{MODEL_NAME}"
    joblib.dump({"model": model, "features": feat_names}, local_path)
    # enviar para GCS
    upload_to_gcs(local_path, GCS_BUCKET, MODEL_NAME)
    print("Model saved and uploaded.")

if __name__ == "__main__":
    main()