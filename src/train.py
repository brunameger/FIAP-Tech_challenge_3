# src/train.py
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIG ---
PROJECT_ID = "cogent-metric-473722-j9"
BQ_TABLE_VIEW = f"{PROJECT_ID}.tc_dataset.vw_ibov_clean"
TARGET = "part"
MODEL_OUT = "artifacts/modelo_ibov.joblib"
PIPE_OUT = "artifacts/pipeline_ibov.joblib"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs("artifacts", exist_ok=True)

# --- BUSCA DADOS DO BIGQUERY ---
client = bigquery.Client(project=PROJECT_ID)
sql = f"""
SELECT cod, asset, type, part, theoricalQty, data_referencia
FROM `{BQ_TABLE_VIEW}`
WHERE part IS NOT NULL
"""
print("Executando query no BigQuery...")
df = client.query(sql).to_dataframe()
print("Linhas obtidas:", len(df))

# --- FEATURE ENGINEERING BASICA ---
# remover linhas sem target (já filtrado) e duplicados
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# transformar data_referencia em features opcionais (ano, mes)
df["data_referencia"] = pd.to_datetime(df["data_referencia"])
df["ref_year"] = df["data_referencia"].dt.year
df["ref_month"] = df["data_referencia"].dt.month
df = df.drop(columns=["data_referencia"])

# escolher colunas
FEATURES = ["asset", "type", "theoricalQty", "ref_year", "ref_month", "cod"]
X = df[FEATURES]
y = df[TARGET].astype(float)

# --- dividir treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Shapes:", X_train.shape, X_test.shape)

# --- PREPROCESSAMENTO ---
# colunas categóricas / numéricas
categorical_cols = ["asset", "type", "cod"]
numeric_cols = ["theoricalQty", "ref_year", "ref_month"]

# transformers
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# --- MODEL PIPELINE ---
model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# --- HYPERPARAMETER SEARCH (RAPIDA) ---
param_dist = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__max_features": ["auto", "sqrt"],
    "model__min_samples_split": [2, 5]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=6,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2
)

print("Iniciando RandomizedSearchCV...")
search.fit(X_train, y_train)

print("Melhor score:", search.best_score_)
print("Melhores params:", search.best_params_)

best_pipeline = search.best_estimator_

# --- AVALIACAO ---
preds = best_pipeline.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print(f"TEST MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# --- SALVAR MODELO E PIPELINE ---
joblib.dump(best_pipeline, PIPE_OUT)
joblib.dump(best_pipeline.named_steps["model"], MODEL_OUT)
print("Modelos salvos em:", PIPE_OUT, MODEL_OUT)

